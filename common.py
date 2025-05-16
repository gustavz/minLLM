import os
import re
import requests
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from typing import Generator, Any, Callable

from config import (
    PROJECT_NAME_KERAS,
    PROJECT_NAME_PYTORCH,
    WANDB_ENTITY,
    DATA_URL,
    DATA_FILE,
    MAX_SEQ_LEN,
    EMBED_DIM,
    NUM_HEADS,
    MLP_DIM,
    NUM_LAYERS,
    DROPOUT_RATE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CLIPNORM,
    TEMPERATURE,
    COMPLETION_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
)


def wandb_init(is_pytorch: bool = False) -> wandb.sdk.wandb_run.Run:
    """Initialize Weights & Biases."""
    project_name = PROJECT_NAME_PYTORCH if is_pytorch else PROJECT_NAME_KERAS
    run = wandb.init(
        project=project_name,
        entity=WANDB_ENTITY,
        config={
            "model_name": f"{project_name}-{MAX_SEQ_LEN}S-{NUM_LAYERS}L-{NUM_HEADS}H-{EMBED_DIM}E-{MLP_DIM}MLP",
            "max_seq_len": MAX_SEQ_LEN,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "embed_dim": EMBED_DIM,
            "mlp_dim": MLP_DIM,
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "clipnorm": CLIPNORM,
            "temperature": TEMPERATURE,
            "completion_length": COMPLETION_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "data_url": DATA_URL,
            "device": DEVICE,
        },
    )
    return run


def tokenize(text: str) -> list[str]:
    """Split text into tokens of lower case words and basic punctuation."""
    return re.findall(r"\b\w+\b|[.,!?;:]", text.lower())


def download_data(url: str = DATA_URL, filename: str = DATA_FILE) -> None:
    """Download the raw text dataset if not already present."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, "w") as f:
            f.write(r.text)
        print("Download complete.")


def load_and_preprocess(
    filename: str = DATA_FILE,
) -> tuple[list[int], list[str], dict[str, int], dict[int, str]]:
    """Load file, tokenize text, and build vocab and ID mappings."""
    text = open(filename, "r").read()
    tokens = tokenize(text)
    vocab = sorted(set(tokens))
    print(f"Text length: {len(text)}")
    print(f"Tokens length: {len(tokens)}")
    print(f"Vocab size: {len(vocab)}")
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    data_ids = [word_to_id[t] for t in tokens]
    return data_ids, vocab, word_to_id, id_to_word


def data_generator(
    data_ids: list[int],
    batch_size: int = BATCH_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    device: torch.device = DEVICE,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield batches of input and shifted labels as torch tensors on global device."""
    num_seq = len(data_ids) // seq_len
    arr = np.array(data_ids[: num_seq * seq_len]).reshape(num_seq, seq_len)
    while True:
        idx = np.random.permutation(num_seq)
        for i in range(0, num_seq, batch_size):
            batch_idx = idx[i : i + batch_size]
            batch = arr[batch_idx]
            inputs = torch.tensor(batch, device=device)
            labels = torch.tensor(np.roll(batch, -1, axis=1), device=device)
            labels[:, -1] = 0
            yield inputs, labels


@torch.no_grad()
def generate_text(
    model: Any,
    prompt: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    is_pytorch_model: bool = True,
    seq_len: int = MAX_SEQ_LEN,
    length: int = COMPLETION_LENGTH,
    temperature: float = TEMPERATURE,
    device: torch.device = DEVICE,
) -> str:
    """
    Generate text of specified length from a prompt using temperature sampling.
    Works with both PyTorch and Keras models.

    Args:
        model: Either a PyTorch nn.Module or a Keras Model
        prompt: The input text prompt
        word_to_id: Dictionary mapping words to IDs
        id_to_word: Dictionary mapping IDs to words
        is_pytorch_model: Whether the model is a PyTorch model (vs Keras)
        seq_len: Maximum sequence length
        length: Length of text to generate
        temperature: Sampling temperature (lower = more deterministic)
        device: The computation device

    Returns:
        Generated text as a string
    """
    if is_pytorch_model:
        model.eval()

    tokens = tokenize(prompt)
    ids = [word_to_id.get(t, 0) for t in tokens][-seq_len:]
    input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    input_ids[0, -len(ids) :] = torch.tensor(ids, device=device)

    generated = []
    for _ in range(length):
        # Forward pass through model
        logits = model(input_ids, training=False)
        # Apply temperature sampling
        next_token_logits = logits[0, -1] / temperature
        # Convert to probabilities with softmax
        probs = F.softmax(next_token_logits, dim=-1)
        # Sample from the probability distribution
        next_id = torch.multinomial(probs, num_samples=1)[0]
        generated.append(next_id.item())
        # Shift input and add new token
        input_ids = torch.cat(
            [input_ids[:, 1:], next_id.unsqueeze(0).unsqueeze(0)], dim=1
        )

    # Convert token IDs back to words
    return " ".join([id_to_word.get(i, "<UNK>") for i in generated])


def run_training_pipeline(
    is_pytorch: bool,
    build_model_fn: Callable[[int], Any],
    train_model_fn: Callable[[Any, Generator, int, str], None],
) -> None:
    """
    Run the full training pipeline for either PyTorch or Keras implementation.

    Args:
        is_pytorch: Whether we're using the PyTorch implementation
        build_model_fn: Function to build the model (takes vocab_size as input)
        train_model_fn: Function to train the model
    """
    print(f"Using device: {DEVICE}")
    download_data()
    run = wandb_init(is_pytorch=is_pytorch)
    data_ids, vocab, word_to_id, id_to_word = load_and_preprocess()
    gen = data_generator(data_ids)

    model = build_model_fn(len(vocab))
    extension = ".pt" if is_pytorch else ".h5"
    weights_file = f"{run.config['model_name']}.weights{extension}"

    if os.path.exists(weights_file):
        print(f"Loading weights from {weights_file}")
        if is_pytorch:
            model.load_state_dict(torch.load(weights_file))
        else:
            model.load_weights(weights_file)
        print("Weights loaded successfully. Continuing training.")

    train_model_fn(model, gen, len(data_ids), weights_file)

    prompts = [
        "BRUTUS:",
        "why is the sky blue?",
        "Upon this present action.",
        "Nor did you think it folly",
    ]

    for prompt in prompts:
        sample = generate_text(
            model, prompt, word_to_id, id_to_word, is_pytorch_model=is_pytorch
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: {sample}")

    run.finish()
