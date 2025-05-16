import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Generator

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
from common import run_training_pipeline


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)

        # Apply causal mask (lower triangular matrix)
        if use_causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Get weighted sum
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Final linear layer
        output = self.out_proj(context)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn = self.attention(x, use_causal_mask=True)
        x = x + attn
        x = self.norm1(x)
        x = self.dropout(x)

        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x


class MinLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = MAX_SEQ_LEN,
        embed_dim: int = EMBED_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        mlp_dim: int = MLP_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Get token embeddings
        token_emb = self.token_embedding(x)

        # Add positional embeddings
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Project back to vocabulary space
        logits = self.output_head(x)
        return logits


def build_model(vocab_size: int) -> nn.Module:
    """Build and prepare the PyTorch model."""
    model = MinLLM(vocab_size=vocab_size)
    model.to(DEVICE)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train_model(
    model: nn.Module,
    data_generator_fn: Generator[tuple[torch.Tensor, torch.Tensor], None, None],
    data_size: int,
    weights_file: str,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    clipnorm: float = CLIPNORM,
) -> None:
    """Train the PyTorch model with custom training loop."""
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # Calculate steps per epoch
    num_seq = data_size // MAX_SEQ_LEN
    steps_per_epoch = max(1, num_seq // BATCH_SIZE)

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 3
    min_delta = 0.0005

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        for step in range(steps_per_epoch):
            # Get batch from generator
            inputs, labels = next(data_generator_fn)

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Calculate accuracy
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == labels).float().mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)

            # Update weights
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # Calculate epoch metrics
        avg_loss = epoch_loss / steps_per_epoch
        avg_acc = epoch_acc / steps_per_epoch

        # Log to Weights & Biases
        wandb.log(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "sparse_categorical_accuracy": avg_acc,
            }
        )

        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")

        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), weights_file)
            print(f"Model saved to {weights_file}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                # Load best weights
                model.load_state_dict(torch.load(weights_file))
                break

        # Learning rate reduction
        if patience_counter > 0 and patience_counter % 2 == 0:
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * 0.5
                print(f"Learning rate reduced to {g['lr']}")
                if g["lr"] < 1e-6:
                    break


def main() -> None:
    """Entry point that uses the common training pipeline."""
    run_training_pipeline(
        is_pytorch=True, build_model_fn=build_model, train_model_fn=train_model
    )


if __name__ == "__main__":
    main()
