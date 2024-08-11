from utilities.text import TextLoaderManager

if __name__ == '__main__':

    TEXT_FOLDER = r'C:\Users\marko\Desktop\data'

    text_loader = TextLoaderManager(folder=TEXT_FOLDER)
    text_loader.read_files()
    texts = text_loader.get_pieces_of_text()
