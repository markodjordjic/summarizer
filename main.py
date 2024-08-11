from utilities.text import TextLoaderManager
from summarizer.summarizer import TextSplitterManager, SummarizerManager

if __name__ == '__main__':

    TEXT_FOLDER = r'C:\Users\marko\Desktop\data'

    text_loader = TextLoaderManager(folder=TEXT_FOLDER)
    text_loader._file_names
    text_loader.read_files()
    texts = text_loader.get_pieces_of_text()

    text_splitter = TextSplitterManager(pieces_of_text=texts)
    text_splitter.split_documents()
    split_documents = text_splitter.get_split_documents()

    summarizer = SummarizerManager(
        chuncked_pieces_of_text=[split_documents[3]]
    )
    summarizer.summarize_text()
    summarized_text = summarizer.get_summarized_text()

