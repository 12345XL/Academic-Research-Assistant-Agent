"""Apply versioned schema migrations without importing corpus or calling models."""
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    from research_agent.settings import Settings
    from research_agent.storage import Repository

    applied = Repository(Settings.from_env()).migrate()
    print("Applied migrations: " + (", ".join(applied) if applied else "none (already current)"))
