import torch

# ===== DATA FILES AND LOCATIONS =====
PROJECT_NAME_KERAS = "min_llm_keras"
PROJECT_NAME_PYTORCH = "min_llm_pytorch"
WANDB_ENTITY = "gustavz"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

# ===== MODEL ARCHITECTURE PARAMETERS =====
MAX_SEQ_LEN = 64  # Maximum sequence length for input context
EMBED_DIM = 128  # Dimension of token embeddings
NUM_HEADS = 4  # Number of attention heads in transformer
MLP_DIM = 512  # Dimension of feed-forward network in transformer (4x embed_dim rule of thumb)
NUM_LAYERS = 3  # Number of transformer layers in the model
DROPOUT_RATE = 0.1  # Dropout rate for regularization

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 32  # Number of sequences per batch
EPOCHS = 50  # Number of training epochs
LEARNING_RATE = 3e-4  # Learning rate for optimizer
WEIGHT_DECAY = 1e-5  # L2 regularization strength
CLIPNORM = 1.0  # Gradient clipping threshold

# ===== TEXT GENERATION PARAMETERS =====
TEMPERATURE = 0.7  # Sampling temperature (lower = more deterministic)
COMPLETION_LENGTH = 100  # Length of generated text completions


# ===== AUTOMATIC DEVICE CONFIGURATION =====
def get_device() -> torch.device:
    """Return the fastest available torch device: MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Set tensor computational device globally
DEVICE = get_device()
