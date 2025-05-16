import os

from common import run_training_pipeline

# Ensure Keras uses the PyTorch backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import wandb
from keras import layers
from typing import Generator, Any

from config import (
    MAX_SEQ_LEN,
    EMBED_DIM,
    NUM_HEADS,
    MLP_DIM,
    NUM_LAYERS,
    DROPOUT_RATE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CLIPNORM,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
)


def transformer_block(
    x: Any,
    num_heads: int = NUM_HEADS,
    embed_dim: int = EMBED_DIM,
    mlp_dim: int = MLP_DIM,
    dropout_rate: float = DROPOUT_RATE,
) -> Any:
    """Build a transformer block with causal attention and feed-forward layers."""
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
    )(x, x, use_causal_mask=True)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dropout(dropout_rate)(x)
    ff = layers.Dense(mlp_dim, activation="gelu")(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(embed_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return x


def build_model(vocab_size: int) -> keras.Model:
    """Build and prepare the Keras model."""
    model = keras.Model(*_build_model_layers(vocab_size))
    model.summary()
    return model


def _build_model_layers(
    vocab_size: int,
    seq_len: int = MAX_SEQ_LEN,
    embed_dim: int = EMBED_DIM,
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    mlp_dim: int = MLP_DIM,
    dropout_rate: float = DROPOUT_RATE,
    device: torch.device = DEVICE,
) -> tuple[Any, Any]:
    """Construct the transformer language model given vocab size and sequence length."""
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    tok_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    tok_emb = layers.Dropout(dropout_rate)(tok_emb)
    pos_emb_layer = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

    def add_pos(x: Any) -> Any:
        T = x.shape[1]
        idx = torch.arange(T, device=device)
        pos = pos_emb_layer(idx)
        return x + torch.unsqueeze(pos, 0)

    x = layers.Lambda(add_pos)(tok_emb)
    for _ in range(num_layers):
        x = transformer_block(
            x,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
        )
    logits = layers.Dense(vocab_size)(x)
    return inputs, logits


def train_model(
    model: keras.Model,
    data_gen: Generator[tuple[torch.Tensor, torch.Tensor], None, None],
    data_size: int,
    weights_file: str,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    clipnorm: float = CLIPNORM,
) -> None:
    """Compile the model and run training for specified epochs."""
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer, loss=loss, metrics=["sparse_categorical_accuracy"]
    )

    model_checkpoint = wandb.keras.WandbModelCheckpoint(
        filepath=weights_file,
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        verbose=1,
    )
    metrics_logger = wandb.keras.WandbMetricsLogger(
        log_freq="epoch",
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=3,
        min_delta=0.0005,
        verbose=1,
        restore_best_weights=True,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=1, min_delta=0.0005, verbose=1, min_lr=1e-6
    )

    # calculate dynamic steps per epoch based on dataset size
    num_seq = data_size // MAX_SEQ_LEN
    steps_per_epoch = max(1, num_seq // BATCH_SIZE)

    model.fit(
        data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint, metrics_logger, early_stopping, reduce_lr],
    )


def main() -> None:
    """Entry point that uses the common training pipeline."""
    run_training_pipeline(
        is_pytorch=False, build_model_fn=build_model, train_model_fn=train_model
    )


if __name__ == "__main__":
    main()
