"""Parse the IAM Dataset."""
import logging
import tarfile
import os
import threading
from typing import Dict, List

import cv2


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
                if archive_name[:5] == 'forms':
                    file.extractall(os.path.join(self.archives_path, 'forms'))
                else:
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

    @staticmethod
    def crop_form(form_id: str) -> None:
        """
        Crop form image.

        :param form_id: Form ID
        """
        image_path = os.path.join('dataset', 'forms', f'{form_id}.png')
        img = cv2.imread(image_path)
        if img is not None:
            img = img[650:2500]
            cv2.imwrite(filename=image_path, img=img)
        else:
            logging.error('Form image not found! Skipping...')

    def parse_dataset(self) -> Dict[str, str]:
        """
        Crop images, match their id with the text on the image.

        :return: Dict:
                - key: Form id.
                - value: Full text on the form.
        """
        parsed_data = {}
        threads = []

        with open(os.path.join(self.archives_path, 'ascii', 'lines.txt'),
                  encoding='UTF8') as lines_file:
            for line in lines_file:
                if self.check_line_correctness(line):
                    line_split = line.split(' ')
                    form_id = '-'.join(line_split[0].split('-')[:2])

                    if parsed_data.get(form_id) is None:
                        threads.append(threading.Thread(target=self.crop_form, args=(form_id,)))

                    value = line_split[-1]. \
                        replace('|', ' '). \
                        replace(' .', '.'). \
                        replace(' ?', '?'). \
                        replace(' ,', ','). \
                        replace(' # ', ' '). \
                        replace(' :', ':'). \
                        strip() + ' '
                    if value[-2] == '-':
                        value = value[:-2]
                    parsed_data.setdefault(form_id, '')
                    parsed_data[form_id] += value

            for i, thread in enumerate(threads):
                thread.start()
                if i % 5 == 0:
                    thread.join()

        return parsed_data


def main() -> None:
    """Unpack archives and parse the dataset."""
    archives_names = ['ascii.tgz', 'formsA-D.tgz', 'formsE-H.tgz', 'formsI-Z.tgz']
    parser = IAMDatasetParser(archives_path='dataset')
    parser.unpack(archives_names)
    parser.parse_dataset()


if __name__ == '__main__':
    main()
