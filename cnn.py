import copy
import os
import time

import numpy as np
import polars as pl
import torch
from torch import nn


class FlareCNN(nn.Module):
    """
    1D convolutional network that scores one window's observation sequence
    (output of sequences.build_sequences()) for the presence of a flare.

    Layout: a stack of Conv1d + ReLU layers along the observation axis, then
    masked max-pooling (each filter's strongest response at any real
    observation; padding and fully masked observations can't contribute),
    then a small dense head that outputs one logit. With three width-3
    convolutions, each pooled response sees up to 7 consecutive observations.

    Arguments
    ---------
    n_channels : int
        Input channels per observation. Defaults to 5 (sequences.CHANNELS).
    n_filters : tuple of int
        Filters in each convolution layer. Defaults to (32, 32, 64).
    kernel_size : int
        Width of each convolution, in observations. Defaults to 3.
    n_hidden : int
        Units in the dense hidden layer. Defaults to 32.
    mask_channel : int
        Index of the mask channel in the input. Defaults to 4.
    """

    def __init__(
        self,
        n_channels=5,
        n_filters=(32, 32, 64),
        kernel_size=3,
        n_hidden=32,
        mask_channel=4,
    ):
        super().__init__()
        self.mask_channel = mask_channel

        layers = []
        in_channels = n_channels
        for out_channels in n_filters:
            # padding keeps the sequence length unchanged
            layers += [
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                ),
                nn.ReLU(),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(in_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, window_sequences):
        """
        Arguments
        ---------
        window_sequences : torch.Tensor
            Shape (batch, n_channels, length).

        Returns
        -------
        logits : torch.Tensor
            Shape (batch,). Apply torch.sigmoid() for a flare probability.
        """
        mask = window_sequences[:, self.mask_channel : self.mask_channel + 1, :] > 0
        responses = self.conv(window_sequences)  # (batch, filters, length)
        responses = responses.masked_fill(~mask, float("-inf"))
        pooled = responses.max(dim=2).values  # (batch, filters)
        # guard: a window with no real observations would give -inf
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return self.head(pooled).squeeze(1)


def mirror(window_sequences, snr_channels=(0, 1)):
    """
    Flips the sign of the SNR channels, turning real flares into dips while
    leaving noise statistically unchanged. Scores on mirrored quiescent
    windows measure the false-positive rate on pure noise, as for the other
    models.

    Arguments
    ---------
    window_sequences : array_like
        Shape (n_rows, n_channels, length).
    snr_channels : tuple of int
        Channels to flip. Defaults to (0, 1) (snr_95, snr_150).

    Returns
    -------
    mirrored : array_like
        Copy of the input with the SNR channels negated.
    """
    mirrored = window_sequences.copy()
    mirrored[:, list(snr_channels)] *= -1
    return mirrored


def star_folds(scores):
    """
    Reads each star's cross-validation fold from the scores table of the
    feature-based models, so the CNN uses exactly the same star-grouped
    folds.

    Arguments
    ---------
    scores : polars.DataFrame
        The saved scores table (data/scores.parquet), with gaia_id and fold.

    Returns
    -------
    folds : dict
        gaia_id -> fold number.
    """
    folds = scores.select("gaia_id", "fold").unique()
    if not folds["gaia_id"].is_unique().all():
        raise ValueError("a star appears in more than one fold")
    return dict(folds.iter_rows())


def validation_stars(train_star_ids, fraction=0.15, seed=100):
    """
    Picks a random subset of the training stars to hold out for early
    stopping. Whole stars are held out, so the validation loss measures
    performance on unseen stars, as the test fold does.

    Arguments
    ---------
    train_star_ids : array_like
        Gaia ids of the stars in the training folds (may repeat).
    fraction : float
        Fraction of the training stars to hold out. Defaults to 0.15.
    seed : int
        Random seed.

    Returns
    -------
    held_out : set
        Gaia ids of the validation stars.
    """
    stars = np.unique(np.asarray(train_star_ids))
    rng = np.random.default_rng(seed)
    n_held_out = max(1, round(fraction * len(stars)))
    return set(rng.choice(stars, n_held_out, replace=False))


def train_model(
    train_sequences,
    train_labels,
    val_sequences,
    val_labels,
    seed=100,
    batch_size=256,
    learning_rate=1e-3,
    max_epochs=100,
    patience=5,
    **model_kwargs,
):
    """
    Trains one FlareCNN with mini-batch Adam on binary cross-entropy, using
    early stopping on the validation loss: training stops once the
    validation loss hasn't improved for `patience` epochs, and the weights
    from the best epoch are returned.

    Arguments
    ---------
    train_sequences, val_sequences : array_like
        Shape (n_rows, n_channels, length), float32.
    train_labels, val_labels : array_like
        0 (noise) or 1 (flare) per row.
    seed : int
        Seed for the starting weights and the batch shuffling.
    batch_size : int
        Windows per gradient step. Defaults to 256.
    learning_rate : float
        Adam learning rate. Defaults to 1e-3.
    max_epochs : int
        Upper limit on passes through the training set. Defaults to 100.
    patience : int
        Epochs without validation improvement before stopping. Defaults to 5.
    **model_kwargs
        Passed to FlareCNN (e.g. n_filters).

    Returns
    -------
    model : FlareCNN
        Trained model with the best-epoch weights, in evaluation mode.
    history : list of dict
        Per epoch: epoch, train_loss, val_loss.
    """
    torch.manual_seed(seed)  # starting weights
    generator = torch.Generator().manual_seed(seed)  # batch order

    model = FlareCNN(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # sigmoid + binary cross-entropy in one numerically stable step
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(train_sequences)
    y_train = torch.from_numpy(np.asarray(train_labels, dtype=np.float32))
    x_val = torch.from_numpy(val_sequences)
    y_val = torch.from_numpy(np.asarray(val_labels, dtype=np.float32))

    best_val_loss = np.inf
    best_state = None
    epochs_since_best = 0
    history = []

    for epoch in range(max_epochs):
        # one epoch: shuffle, then one Adam step per batch
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()  # clear the previous step's gradients
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()  # gradient of the loss for every weight
            optimizer.step()  # move the weights downhill
            loss_sum += loss.item() * len(batch)
        train_loss = loss_sum / len(order)

        # validation loss with the weights at the end of this epoch
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict(model, window_sequences, batch_size=4096):
    """
    Flare probabilities from a trained model.

    Arguments
    ---------
    model : FlareCNN
        Trained model.
    window_sequences : array_like
        Shape (n_rows, n_channels, length), float32.
    batch_size : int
        Rows per forward pass (memory only; doesn't change the result).

    Returns
    -------
    probabilities : array_like
        Shape (n_rows,), between 0 and 1.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(window_sequences), batch_size):
            batch = torch.from_numpy(window_sequences[start : start + batch_size])
            outputs.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(outputs)


def cross_validate(
    window_sequences,
    sequence_rows,
    seed=100,
    val_fraction=0.15,
    model_dir=None,
    **train_kwargs,
):
    """
    Out-of-fold CNN scores with the same star-grouped folds as the
    feature-based models. For each fold: hold the fold out as the test set,
    split the remaining stars into training and validation stars, train with
    early stopping, then score the test fold once. Quiescent windows of the
    test fold are also scored mirrored (SNR signs flipped), for the
    noise-based thresholds.

    Arguments
    ---------
    window_sequences : array_like
        Output of sequences.build_sequences(), shape (n_rows, 5, length).
    sequence_rows : polars.DataFrame
        One row per sequence, in the same order, with gaia_id, window_start,
        window_id, inj_id, source, label and fold.
    seed : int
        Seed for training and for the choice of validation stars.
    val_fraction : float
        Fraction of the training stars held out for early stopping.
    model_dir : str, optional
        If given, each fold's trained weights are saved there as
        seed{seed}_fold{fold}.pt, so new windows (e.g. real events) can be
        scored later by the model that never saw their star (load_model()).
    **train_kwargs
        Passed to train_model() (e.g. batch_size, n_filters).

    Returns
    -------
    cnn_scores : polars.DataFrame
        Keys (gaia_id, window_start, window_id, inj_id, source, label, fold),
        cnn_score (out-of-fold flare probability) and cnn_mirrored_score
        (null for injections).
    history : polars.DataFrame
        Per fold and epoch: fold, epoch, train_loss, val_loss.
    """
    labels = sequence_rows["label"].to_numpy()
    folds = sequence_rows["fold"].to_numpy()
    star_ids = sequence_rows["gaia_id"].to_numpy()
    is_quiescent = (sequence_rows["source"] == "quiescent").to_numpy()

    cnn_score = np.full(sequence_rows.height, np.nan)
    cnn_mirrored_score = np.full(sequence_rows.height, np.nan)
    histories = []

    for fold in np.unique(folds):
        fold_start = time.perf_counter()
        test = folds == fold
        held_out = validation_stars(star_ids[~test], val_fraction, seed)
        val = ~test & np.isin(star_ids, list(held_out))
        train = ~test & ~val

        model, history = train_model(
            window_sequences[train],
            labels[train],
            window_sequences[val],
            labels[val],
            seed=seed,
            **train_kwargs,
        )
        cnn_score[test] = predict(model, window_sequences[test])
        test_quiescent = test & is_quiescent
        cnn_mirrored_score[test_quiescent] = predict(
            model, mirror(window_sequences[test_quiescent])
        )
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_dir}/seed{seed}_fold{fold}.pt")

        histories += [{"fold": int(fold), **h} for h in history]
        best = min(history, key=lambda h: h["val_loss"])
        print(
            f"fold {fold}: {len(history)} epochs, best epoch {best['epoch']}, "
            f"val loss {best['val_loss']:.4f}, N_train {train.sum()}, N_val {val.sum()}, "
            f"{time.perf_counter() - fold_start:.1f} s"
        )

    cnn_scores = sequence_rows.select(
        "gaia_id", "window_start", "window_id", "inj_id", "source", "label", "fold"
    ).with_columns(
        pl.Series("cnn_score", cnn_score),
        pl.Series("cnn_mirrored_score", cnn_mirrored_score).fill_nan(None),
    )
    return cnn_scores, pl.DataFrame(histories)


def load_model(model_dir, seed, fold, **model_kwargs):
    """
    Loads one fold's trained weights saved by cross_validate(model_dir=...).

    Arguments
    ---------
    model_dir : str
        Folder the weights were saved in.
    seed, fold : int
        Which training run and fold to load.
    **model_kwargs
        Must match the FlareCNN settings used in training (defaults if none
        were passed).

    Returns
    -------
    model : FlareCNN
        Model in evaluation mode.
    """
    model = FlareCNN(**model_kwargs)
    model.load_state_dict(torch.load(f"{model_dir}/seed{seed}_fold{fold}.pt"))
    model.eval()
    return model


def ensemble_predict(
    window_sequences, star_ids, folds, model_dir, seeds, **model_kwargs
):
    """
    Scores new sequences with the saved models: each row is scored by the
    fold model that never saw its star, for every seed, and the seeds'
    probabilities are averaged (the same ensemble as the out-of-fold scores).

    Arguments
    ---------
    window_sequences : array_like
        Shape (n_rows, n_channels, length).
    star_ids : array_like
        Gaia id of each row.
    folds : dict
        gaia_id -> fold number (output of star_folds()).
    model_dir : str
        Folder with the saved weights.
    seeds : list of int
        Training seeds to average over.
    **model_kwargs
        FlareCNN settings used in training.

    Returns
    -------
    scores : array_like
        Shape (n_rows,), mean flare probability over the seeds.
    """
    row_folds = np.array([folds[g] for g in star_ids])
    scores = np.zeros(len(window_sequences))
    for seed in seeds:
        for fold in np.unique(row_folds):
            in_fold = row_folds == fold
            model = load_model(model_dir, seed, fold, **model_kwargs)
            scores[in_fold] += predict(model, window_sequences[in_fold])
    return scores / len(seeds)
