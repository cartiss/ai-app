import logging
from pathlib import Path
from zipfile import ZipFile


class DataDownloader:
    """Data handler."""

    @staticmethod
    def _kaggle_authenticate():
        """Authenticate to Kaggle API."""
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        return api

    def download_kaggle_dataset(
        self,
        kaggle_id: str,
        archive_name: str,
        extract_path: Path,
        is_competition: bool = False,
    ) -> None:
        """
        Downloads and unpacks a Kaggle dataset into the specified directory.

        :param kaggle_id: Kaggle dataset/competition ID
        :param archive_name: Archive name to extract
        :param extract_path: Path to extract archive
        :param is_competition: False if kaggle dataset, otherwise - kaggle competition
        """
        api = self._kaggle_authenticate()

        if not is_competition:
            api.dataset_download_files(kaggle_id)
        else:
            api.competition_download_files(kaggle_id)

        self._extract_dataset(archive_name=archive_name, dataset_path=extract_path)

    @staticmethod
    def _extract_dataset(archive_name: str, dataset_path: Path) -> None:
        """
        Extract dataset archive.

        :param archive_name: Archive name
        :param dataset_path: Path to the folder to extract the dataset into
        """
        try:
            zf = ZipFile(archive_name)
        except FileNotFoundError as error:
            logging.error(f'Cannot find archive with name: {archive_name}!')
            raise error

        if not dataset_path.is_dir():
            Path.mkdir(dataset_path)

        zf.extractall(path=dataset_path)
        zf.close()
