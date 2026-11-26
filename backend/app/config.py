# Centralized configuration for the FinMind application

# Currency used across the demo (hard‑coded as per locked decision)
CURRENCY = "MYR"

# ==============================================================
# Machine‑learning configuration (Phase 6)
# ==============================================================
# Directory where trained models and scalers are persisted (outside source control)
MODEL_STORAGE_PATH = "backend/model_store"

# Forecasting hyper‑parameters – read from environment variables with sensible defaults
import os

FORECAST_LOOKBACK_DAYS = int(os.getenv("FORECAST_LOOKBACK_DAYS", "30"))
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "14"))
FORECAST_EPOCHS = int(os.getenv("FORECAST_EPOCHS", "20"))
FORECAST_BATCH_SIZE = int(os.getenv("FORECAST_BATCH_SIZE", "16"))
FORECAST_LEARNING_RATE = float(os.getenv("FORECAST_LEARNING_RATE", "0.001"))

# Random seed for reproducibility of deterministic pipelines
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Embedding configuration for Document Intelligence (Phase 7)
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'openai')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', '')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))  # characters
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
VECTOR_TOP_K = int(os.getenv('VECTOR_TOP_K', '5'))

# AI Provider Configuration (Block 5)
AI_PROVIDER_MODE = os.getenv("AI_PROVIDER_MODE", "auto")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Ordered fallback models tried when the primary model is unavailable
# (capacity 503/429, or the primary is rotated out). All verified working.
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.8-flash,gemini-flash-latest,gemini-flash-lite-latest",
    ).split(",")
    if m.strip()
]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

