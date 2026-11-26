import ssl
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import certifi


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_dataset() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    zip_path = project_dir / "ml-latest-small.zip"

    data_dir.mkdir(parents=True, exist_ok=True)

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(DATASET_URL, context=ssl_context) as response, zip_path.open("wb") as zip_file:
        zip_file.write(response.read())

    with ZipFile(zip_path) as archive:
        archive.extractall(data_dir)

    extracted_dir = data_dir / "ml-latest-small"
    if extracted_dir.is_dir():
        for source in extracted_dir.iterdir():
            source.replace(data_dir / source.name)
        extracted_dir.rmdir()

    zip_path.unlink()


if __name__ == "__main__":
    download_dataset()
