import os

# Ensure Keras uses the PyTorch backend before importing keras
os.environ['KERAS_BACKEND'] = 'torch'

import re
import requests
import numpy as np
import torch
import keras
import wandb
from keras import layers
from typing import List, Dict, Tuple, Generator, Any

# ===== DATA FILES AND LOCATIONS =====
PROJECT_NAME = "min_llm_keras_torch"
WANDB_ENTITY = "gustavz"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

# ===== MODEL ARCHITECTURE PARAMETERS =====
MAX_SEQ_LEN = 64                         # Maximum sequence length for input context
EMBED_DIM = 128                          # Dimension of token embeddings
NUM_HEADS = 4                            # Number of attention heads in transformer 
MLP_DIM = 512                            # Dimension of feed-forward network in transformer (4x embed_dim rule of thumb)
NUM_LAYERS = 3                           # Number of transformer layers in the model 
DROPOUT_RATE = 0.1                       # Dropout rate for regularization

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 32                          # Number of sequences per batch 
EPOCHS = 50                              # Number of training epochs 
LEARNING_RATE = 3e-4                     # Learning rate for optimizer 
WEIGHT_DECAY = 1e-5                      # L2 regularization strength
CLIPNORM = 1.0                           # Gradient clipping threshold

# ===== TEXT GENERATION PARAMETERS =====
TEMPERATURE = 0.7                        # Sampling temperature (lower = more deterministic)
COMPLETION_LENGTH = 100                  # Length of generated text completions

def get_device() -> torch.device:
    """Return the fastest available torch device: MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Set tensor computational device globally
DEVICE = get_device()

def wandb_init() -> wandb.sdk.wandb_run.Run:
    """Initialize Weights & Biases."""
    run = wandb.init(
        project=PROJECT_NAME,
        entity=WANDB_ENTITY,
        config={
            "model_name": f"{PROJECT_NAME}-{MAX_SEQ_LEN}S-{NUM_LAYERS}L-{NUM_HEADS}H-{EMBED_DIM}E-{MLP_DIM}MLP",
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
        }
    )
    return run


def tokenize(text: str) -> List[str]:
    """Split text into tokens of lower case words and basic punctuation."""
    return re.findall(r'\b\w+\b|[.,!?;:]', text.lower())

def download_data(url: str = DATA_URL, filename: str = DATA_FILE) -> None:
    """Download the raw text dataset if not already present."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'w') as f:
            f.write(r.text)
        print("Download complete.")

def load_and_preprocess(filename: str = DATA_FILE) -> Tuple[List[int], List[str], Dict[str, int], Dict[int, str]]:
    """Load file, tokenize text, and build vocab and ID mappings."""
    text = open(filename, 'r').read()
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
    data_ids: List[int], 
    batch_size: int = BATCH_SIZE, 
    seq_len: int = MAX_SEQ_LEN, 
    device: torch.device = DEVICE
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield batches of input and shifted labels as torch tensors on global device."""
    num_seq = len(data_ids) // seq_len
    arr = np.array(data_ids[:num_seq * seq_len]).reshape(num_seq, seq_len)
    while True:
        idx = np.random.permutation(num_seq)
        for i in range(0, num_seq, batch_size):
            batch_idx = idx[i:i + batch_size]
            batch = arr[batch_idx]
            inputs = torch.tensor(batch, device=device)
            labels = torch.tensor(np.roll(batch, -1, axis=1), device=device)
            labels[:, -1] = 0
            yield inputs, labels

def transformer_block(
    x: Any, 
    num_heads: int = NUM_HEADS, 
    embed_dim: int = EMBED_DIM, 
    mlp_dim: int = MLP_DIM,
    dropout_rate: float = DROPOUT_RATE
) -> Any:
    """Build a transformer block with causal attention and feed-forward layers."""
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate
    )(x, x, use_causal_mask=True)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dropout(dropout_rate)(x)
    ff = layers.Dense(mlp_dim, activation='gelu')(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(embed_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return x

def build_model(
    vocab_size: int, 
    seq_len: int = MAX_SEQ_LEN, 
    embed_dim: int = EMBED_DIM, 
    num_layers: int = NUM_LAYERS, 
    num_heads: int = NUM_HEADS, 
    mlp_dim: int = MLP_DIM, 
    dropout_rate: float = DROPOUT_RATE,
    device: torch.device = DEVICE
) -> keras.Model:
    """Construct the transformer language model given vocab size and sequence length."""
    inputs = layers.Input(shape=(seq_len,), dtype='int32')
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
        x = transformer_block(x, num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim, dropout_rate=dropout_rate)
    logits = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs, logits)
    model.summary()
    return model

def train_model(
    model: keras.Model, 
    data_gen: Generator[Tuple[torch.Tensor, torch.Tensor], None, None], 
    data_size: int,
    weights_file: str,
    epochs: int = EPOCHS, 
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    clipnorm: float = CLIPNORM,
) -> None:
    """Compile the model and run training for specified epochs."""
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

    model_checkpoint = wandb.keras.WandbModelCheckpoint(
        filepath=weights_file,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss',
        verbose=1
    )
    metrics_logger = wandb.keras.WandbMetricsLogger(
        log_freq="epoch",
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        min_delta=0.0005,
        verbose=1,
        restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
        min_delta=0.0005,
        verbose=1,
        min_lr=1e-6
    )
    
    # calculate dynamic steps per epoch based on dataset size
    num_seq = data_size // MAX_SEQ_LEN
    steps_per_epoch = max(1, num_seq // BATCH_SIZE)
    
    model.fit(
        data_gen, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs,
        callbacks=[model_checkpoint, metrics_logger, early_stopping, reduce_lr]
    )

@torch.no_grad()
def generate_text(
    model: keras.Model, 
    prompt: str, 
    word_to_id: Dict[str, int], 
    id_to_word: Dict[int, str], 
    seq_len: int = MAX_SEQ_LEN, 
    length: int = COMPLETION_LENGTH,
    temperature: float = TEMPERATURE,
    device: torch.device = DEVICE
) -> str:
    """Generate text of specified length from a prompt using temperature sampling."""
    tokens = tokenize(prompt)
    ids = [word_to_id.get(t, 0) for t in tokens][-seq_len:]
    input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    input_ids[0, -len(ids):] = torch.tensor(ids, device=device)
    generated = []
    for _ in range(length):
        logits = model(input_ids, training=False)
        # Apply temperature sampling
        next_token_logits = logits[0, -1] / temperature
        # Convert to probabilities with softmax
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        # Sample from the probability distribution
        next_id = torch.multinomial(probs, num_samples=1)[0]
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids[:, 1:], next_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return ' '.join([id_to_word.get(i, "<UNK>") for i in generated])

def main() -> None:
    """Orchestrate data download, training, and text generation pipeline."""
    print(f"Using device: {DEVICE}")
    download_data()
    run = wandb_init()
    data_ids, vocab, word_to_id, id_to_word = load_and_preprocess()
    gen = data_generator(data_ids)
    model = build_model(len(vocab))
    
    weights_file = f"{run.config['model_name']}.weights.h5"
    if os.path.exists(weights_file):
        print(f"Loading weights from {weights_file}")
        model.load_weights(weights_file)
        print("Weights loaded successfully. Continuing training.")
    
    train_model(model, gen, len(data_ids), weights_file)
    
    prompts = [
        "BRUTUS:",
        "why is the sky blue?",
        "Upon this present action.",
        "Nor did you think it folly",
    ]
    
    for prompt in prompts:
        sample = generate_text(model, prompt, word_to_id, id_to_word)
        print(f"\nPrompt: '{prompt}'")
        print(f"Completion: {sample}")
    
    run.finish()

if __name__ == "__main__":
    main()
