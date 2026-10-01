"""
Where model weights live.

These are plain path constants, kept apart from llm_loader so that reading
them does not import llama_cpp.  preflight.py and prefetch_models.py need the
paths but not the engine: preflight's whole job is to report that the engine
is missing, and prefetching weights is the step you want to run *before*
building it.  Importing the engine to learn a directory name made both of
them die on the condition they exist to handle.
"""

import os

# Settable because a pod's container disk is ephemeral and far too small for
# the ~145 GB model set: point this at the persistent volume (setup_runpod.sh
# does) or the whole download is lost when the pod stops.  Note that HF_HOME
# alone cannot do this -- hf_hub_download is called with an explicit
# cache_dir, which takes precedence over HF_HOME.
DEFAULT_WEIGHTS_DIR = os.environ.get("MODEL_WEIGHTS_DIR") or os.path.join("models", "weights")

# Locally built GGUF variants land here.  Checking this directory first means
# a built variant is used through its normal registry key, so run_phase.py and
# the analysis need no special casing.
BUILT_DIR = os.path.join(DEFAULT_WEIGHTS_DIR, "built")
