from pathlib import Path
from zipfile import ZipFile


class DataHandler:
    """Data Handler."""

    @staticmethod
    def kaggle_authenticate():
        """Authenticate to Kaggle API."""
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        return api

    @staticmethod
    def extract_dataset(archive_name: str, dataset_path: Path) -> None:
        """
        Extract dataset archive.

        :param archive_name: Archive name
        :param dataset_path: Path to the folder to extract the dataset into
        """
        zf = ZipFile(archive_name)

        if not dataset_path.is_dir():
            Path.mk_dir(dataset_path)

        zf.extractall(path=dataset_path)
        zf.close()

    @staticmethod
    def download_kaggle_dataset(kaggle_api, dataset_id: str) -> None:
        """
        Download and extract Kaggle dataset by ID.

        :param kaggle_api: Kaggle API object
        :param dataset_id: Dataset ID
        """
        kaggle_api.dataset_download_files(dataset_id)

    @staticmethod
    def download_kaggle_competition(kaggle_api, competition_id: str) -> None:
        """
        Download and extract Kaggle competition data by competition ID.

        :param kaggle_api: Kaggle API object
        :param competition_id: Competition ID
        """
        kaggle_api.competition_download_files(competition_id)
