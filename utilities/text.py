import os
from pathlib import Path

class TextLoader:

    def __init__(self) -> None:
        pass


class TextLoaderManager:

    def __init__(self, folder: str = None) -> None:
        self.folder = Path(folder)
        self._file_names: list [str] = []
        self._pieces_of_text: list[str] = []

    def _get_file_names_from_folder(self):
        
        assert self.folder is not None, 'No folder to read the data.'

        self._file_names = [
            file for file in os.listdir(self.folder)
            if os.path.isfile(os.path.join(self.folder, file))
        ]

    @staticmethod
    def _read_file(complete_file_name: Path = None):
        with open(complete_file_name, 'r', encoding='utf8') as text_file:
            content = text_file.read()

        return content

    def _read_files(self):
        
        assert self._file_names is not None, 'No files to read the data.'

        for file_name in self._file_names:
            complete_file_name = Path(self.folder, file_name)
            content = self._read_file(complete_file_name=complete_file_name)
            self._pieces_of_text.extend([content])

    def read_files(self):
        self._get_file_names_from_folder()
        self._read_files()

    def get_pieces_of_text(self):

        assert len(self._pieces_of_text) > 0, 'No text has been read.'

        return self._pieces_of_text