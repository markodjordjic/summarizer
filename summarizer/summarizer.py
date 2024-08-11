from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Summarizer:

    def __init__(self) -> None:
        pass

class SummarizerManager:

    def __init__(self, chuncked_pieces_of_text) -> None:

        self.chuncked_pieces_of_text
        


class TextSplitter:

    def __init__(self, 
                 chunk_size: int = 384, 
                 document: str = None) -> None:
        self.chunk_size = chunk_size
        self.document = document
        self._splitter = None
        self._converted_document = None

    def _instantiate_splitter(self):
        self._splitter = RecursiveCharacterTextSplitter(
            separators=['\n'],
            chunk_size=self.chunk_size, 
            chunk_overlap=0
        )
    def _split_document(self):
        self._converted_document = self._splitter.create_documents(
            [self.document]
        )

    def split_document(self):
        
        assert len(self.document) > 0, 'Document size 0.'
        
        self._instantiate_splitter()
        self._split_document()

    def get_converted_document(self):

        assert self._converted_document is not None, 'No split document.'
    
        return self._converted_document

class TextSplitterManager:

    def __init__(self, pieces_of_text: list[str] = None) -> None:
        self.pieces_of_text = pieces_of_text
        self._split_documents: list[str] = []

    @staticmethod
    def _split_document(text):
        text_splitter = TextSplitter(document=text)
        text_splitter.split_document()
        converted_text = text_splitter.get_converted_document()
        print(
            f'Text with: {len(text)} characters, split into {len(converted_text)} pieces.'
        )

        return converted_text
    
    def split_documents(self):

        assert len(self.pieces_of_text) > 0, 'No documents to split.' 

        for text in self.pieces_of_text:
            converted_text = self._split_document(text=text)
            self._split_documents.extend([converted_text])

    def get_split_documents(self):

        assert len(self._split_documents) > 0, 'No split documents.'

        return self._split_documents
