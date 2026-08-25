"""Download the M5 Forecasting (Accuracy) dataset from Kaggle.

Usage:
    python -m demandcast.data.download
"""

from __future__ import annotations

import logging
import zipfile

from demandcast.settings import get_config, resolve_path

logger = logging.getLogger(__name__)

COMPETITION = "m5-forecasting-accuracy"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import kaggle

    raw_dir = resolve_path(get_config()["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle competition %s", COMPETITION)
    kaggle.api.competition_download_files(COMPETITION, path=str(raw_dir))
    for zf in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(raw_dir)
        zf.unlink()
    logger.info("Raw files: %s", sorted(p.name for p in raw_dir.iterdir()))


if __name__ == "__main__":
    main()
