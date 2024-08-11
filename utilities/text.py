from pathlib import Path

class TextLoader:

    def __init__(self) -> None:
        pass


class TextLoaderManager:

    def __init__(self, folder: str = None) -> None:
        self.folder = Path(folder)
        self._file_names: list [str] = None
        self._pieces_of_text: list[str] = None

    def _get_files_from_folder(self):
        
        assert self.folder is not None, 'No folder to read the data.'

    def _read_files(self):
        pass

    def get_pieces_of_text(self):

        assert self._pieces_of_text is not None, 'No text has been read.'

        return self._pieces_of_text