from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
LOCAL_ENV_PATH = BASE_DIR / ".env"
LEGACY_ENV_PATH = BASE_DIR.parent / "auto_Youtube" / "shorts" / ".env"


def project_env_paths() -> tuple[Path, ...]:
    return (LOCAL_ENV_PATH, LEGACY_ENV_PATH)


def load_project_env() -> None:
    for path in project_env_paths():
        load_dotenv(path, override=False)


def format_checked_env_paths() -> str:
    return ", ".join(str(path) for path in project_env_paths())
