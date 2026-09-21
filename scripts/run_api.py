"""Start the PostgreSQL/S3 workbench API on localhost (no silent file fallback)."""
from pathlib import Path
import sys

from dotenv import load_dotenv
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    from research_agent.settings import Settings
    Settings.from_env()  # Fail clearly before accepting requests if setup is missing.
    uvicorn.run("research_agent.api:app", host="127.0.0.1", port=8011)
