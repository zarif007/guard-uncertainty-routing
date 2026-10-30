import os

DEFAULT_WEIGHTS_DIR = os.environ.get("MODEL_WEIGHTS_DIR") or os.path.join("models", "weights")

BUILT_DIR = os.path.join(DEFAULT_WEIGHTS_DIR, "built")
