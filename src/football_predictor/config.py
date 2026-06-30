import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    db_path: Path = Path(os.getenv("FOOTBALL_PREDICTOR_DB_PATH", "data/football_predictor.sqlite"))


settings = Settings()
