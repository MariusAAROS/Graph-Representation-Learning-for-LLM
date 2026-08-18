import re


def model_slug(model_name: str) -> str:
    """HF repo id -> run-name tag, e.g. "Qwen/Qwen2.5-1.5B" -> "qwen2.5-1.5b"."""
    base = model_name.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", base).strip("-") or "model"
