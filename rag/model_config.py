import os
from pathlib import Path


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PROJECT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def _hub_cache_dir() -> Path:
    """Tim thu muc cache Hugging Face theo dung thu tu uu tien cua library."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_snapshot_path(model_name: str = MODEL_NAME) -> Path | None:
    """Tra ve snapshot local da duoc Hugging Face cache, neu co."""
    repo_cache = _hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    ref_file = repo_cache / "refs" / "main"
    if not ref_file.exists():
        return None

    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot_path = repo_cache / "snapshots" / revision
    if snapshot_path.exists():
        return snapshot_path

    return None


def _is_model_dir(path: Path) -> bool:
    """Kiem tra toi thieu de tranh chon nham thu muc model tai do dang."""
    return (
        path.exists()
        and (path / "modules.json").exists()
        and (path / "config_sentence_transformers.json").exists()
    )


def resolve_local_model_path() -> Path:
    """Chi tra ve duong dan local; runtime khong duoc phu thuoc internet."""
    if _is_model_dir(PROJECT_MODEL_DIR):
        return PROJECT_MODEL_DIR

    snapshot_path = _cached_snapshot_path()
    if snapshot_path is not None and _is_model_dir(snapshot_path):
        return snapshot_path

    raise FileNotFoundError(
        "Chua tim thay model embedding trong may. Hay chay "
        "`venv\\Scripts\\python.exe preload_model.py` khi co internet truoc."
    )
