"""Central configuration: paths, model names, and environment checks.

Everything the pipeline needs to know about *settings* lives here, so the
other modules stay focused on behavior.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

_config_path = PROJECT_ROOT / "config.py"

if not _config_path.exists():
    raise FileNotFoundError(
        f"config.py not found: {_config_path}"
    )

_config_spec = importlib.util.spec_from_file_location(
    "app_config", _config_path
)

if _config_spec is None or _config_spec.loader is None:
    raise ImportError("Unable to load config.py")

_config_module = importlib.util.module_from_spec(_config_spec)

sys.modules["app_config"] = _config_module

_config_spec.loader.exec_module(_config_module)

MODE_LABELS = _config_module.MODE_LABELS
groq_api_key = _config_module.groq_api_key
load_environment = _config_module.load_environment
