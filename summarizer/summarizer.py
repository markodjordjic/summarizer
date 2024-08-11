from copy import deepcopy
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utilities.general import environment_reader

environment = environment_reader('./.env')

class Summarizer:

    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        api_key=environment['OPENAI_API_KEY'],
        temperature=0
    )

    template = """You are an summarization assistant. Your job is
        to summarize two pieces of text into a single piece of text.
        Pay attention to people names in the original pieces of text,
        when summarizing them. Here is the first piece of text 
        {first_text} and here is the second piece of text {second_text}.
        Make your summary as a text is complete on its own. Final 
        summary cannot have more than four paragraphs. Each paragraph 
        cannot be longer than 448 characters including white spaces. 
        Terminate each paragraph with `\n\n`.
    """

    prompt = PromptTemplate.from_template(template)
    
    map_chain = LLMChain(llm=llm, prompt=prompt)

    def __init__(self, chuncked_piece_of_text) -> None:
        self.chucked_piece_of_text = chuncked_piece_of_text
        self._summarized_text = deepcopy(chuncked_piece_of_text)
        self._summarization_pairs = None

    @staticmethod
    def _create_summarization_pairs(pairs):
        first_list = list(range(0, len(pairs))) 
        second_list = []
        for i in range(0, len(first_list)-1, 2):
            if first_list[i]+1 == first_list[i+1]:
                second_list.append([first_list[i], first_list[i+1]])

        return second_list
    
    @staticmethod
    def _log_message(cycle, index):
        if index%16 == 0:
            print(f'Summary cycle: {cycle}, pair: {index}')
    
    def _get_pieces_of_text(self, pair):

        return self._summarized_text[pair[0]], self._summarized_text[pair[1]]


    def _summarize_single_pair(self, pair: list = []) -> Document:
        first_piece_of_text, second_piece_of_text = \
            self._get_pieces_of_text(pair)        
        result = self.map_chain.invoke({
            'first_text': first_piece_of_text,
            'second_text': second_piece_of_text
        })
        document = Document(page_content=result['text'])

        return document

    def _summarize_pairs_within_cycle(self, cycle_index) -> list[str]:
        new_texts = []
        for pair_index, pair in enumerate(self._summarization_pairs):
            self._log_message(cycle=cycle_index, index=pair_index)
            document = self._summarize_single_pair(pair=pair)
            new_texts.extend([document])

        return new_texts

    def _summarize(self):
        cycle = 1
        while len(self._summarized_text) > 1:
            self._summarization_pairs = \
                self._create_summarization_pairs(pairs=self._summarized_text)
            summary_cycle = \
                self._summarize_pairs_within_cycle(cycle_index=cycle)
            cycle = cycle + 1
            self._summarized_text = summary_cycle
        
        print('summarization complete')       

    def summarize(self):
        self._summarize()

    def get_summarized_text(self):

        return self._summarized_text

class SummarizationManager:

    def __init__(self, chuncked_pieces_of_text: list = None) -> None:
        self.chuncked_pieces_of_text = chuncked_pieces_of_text
        self._summarized_text = []

    def summarize_text(self):
        for chuncked_text in self.chuncked_pieces_of_text:
            summarizer = Summarizer(chuncked_piece_of_text=chuncked_text)
            summarizer.summarize()            

    def get_summarized_text(self):
        
        assert self._summarized_text is not None, 'No summarized text.'

        return self._summarized_text
    


class TextSplitter:

    def __init__(self, 
                 chunk_size: int = 448, 
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
