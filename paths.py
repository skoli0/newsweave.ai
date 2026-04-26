"""Project paths - absolute so the DB is the same no matter the process cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = str(DATA_DIR / "newsweave.db")
