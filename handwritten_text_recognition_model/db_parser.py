"""Parse the IAM Dataset."""

import tarfile
import os
from typing import Dict, List


class BaseDatasetParser:
    """Base Dataset Parser."""

    def __init__(self, archives_path: str) -> None:
        """Dataset Parser initialization."""
        self.archives_path = archives_path

    def unpack(self, archives_names: List[str]) -> None:
        """
        Unpack specified archives.

        :param archives_names: Names of the archives to be unpacked.
        """
        for archive_name in archives_names:
            with tarfile.open(os.path.join(self.archives_path, archive_name), mode='r|gz') as file:
                file.extractall(os.path.join(self.archives_path, archive_name.split('.')[0]))

    def parse_dataset(self) -> Dict[str, str]:
        """
        Match image id with the text on the image.

        :return: Dict:
                - key: Page id.
                - value: Full text on the page.
        """
        pass  # pylint: disable=unnecessary-pass


class IAMDatasetParser(BaseDatasetParser):
    """IAM Handwriting Dataset Parser."""

    @staticmethod
    def check_line_correctness(line: str) -> bool:
        """
        Check if the line is empty or commented out.

        :param line: Text line to check.
        :return: True if the line is correct.
                 False if the line is empty or commented out.
        """
        if line[0] == '#' or line.strip() == '':
            return False
        return True

    def parse_dataset(self) -> Dict[str, str]:
        """
        Match image id with the text on the image.

        :return: Dict:
                - key: Page id.
                - value: Full text on the page.
        """
        tmp_dict = {}

        with open(os.path.join(self.archives_path, 'ascii', 'lines.txt'),
                  encoding='UTF8') as lines_file:
            for line in lines_file:
                if self.check_line_correctness(line):
                    line_split = line.split(' ')
                    key = '-'.join(line_split[0].split('-')[:2])
                    value = line_split[-1].\
                        replace('|', ' ').\
                        replace(' .', '.'). \
                        replace(' ?', '?').\
                        replace(' ,', ',').\
                        replace(' # ', ' ').\
                        replace(' :', ':').\
                        strip() + ' '
                    if value[-2] == '-':
                        value = value[:-2]
                    tmp_dict.setdefault(key, '')
                    tmp_dict[key] += value

        return tmp_dict


def main() -> None:
    """Unpack archives and parse the dataset."""
    archives_names = ['ascii.tgz', 'formsE-H.tgz', 'formsI-Z.tgz']
    parser = IAMDatasetParser(archives_path='dataset')
    parser.unpack(archives_names)
    parser.parse_dataset()


if __name__ == '__main__':
    main()
