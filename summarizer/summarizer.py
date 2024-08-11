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

    map_template = """You are an summarization assistant. Your job is
        to summarize two pieces of text into a single piece of text.
        Pay attention to people names in the original pieces of text,
        when summarizing them. There is the first piece of text {first_text}
        and the second piece of text {second_text}. Summarize them in 
        such way that the final summary has no more than four paragraphs 
        and that each paragraph is not greater than 392 characters. 
        Separate each paragraph with `\n\n`.
    """

    map_prompt = PromptTemplate.from_template(map_template)
    
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    def __init__(self, chuncked_piece_of_text) -> None:
        self.chucked_piece_of_text = chuncked_piece_of_text
        self._summarized_text = deepcopy(chuncked_piece_of_text)
        self._summarization_pairs = None

    @staticmethod
    def _create_summarization_pairs(pairs):
        pairs_count = len(pairs)

        first_list = list(range(0, pairs_count)) 
        second_list = []
        for i in range(0,len(first_list)-1,2):
            if first_list[i]+1==first_list[i+1]:
                second_list.append([first_list[i],first_list[i+1]])

        return second_list

    def _summaize_single_pair(self):
        pass

    def _summarize_all_remaining_pairs(self):
        i = 1
        while len(self._summarized_text) > 1:
            new_texts = []
            self._summarization_pairs = self._create_summarization_pairs(
                pairs=self._summarized_text
            )
            for index, pair in enumerate(self._summarization_pairs):
                if index%10 == 0:
                    print(f'Summary cycle: {i}; pair{index}')
                first_piece_of_text = self._summarized_text[pair[0]]
                second_piece_of_text = self._summarized_text[pair[1]]
                result = self.map_chain.invoke(
                    {
                        'first_text': first_piece_of_text,
                        'second_text': second_piece_of_text
                    }
                )
                document = Document(page_content=result['text'])
                new_texts.extend([document])
            i = i + 1
            self._summarized_text = new_texts
        
        print('summarization complete')       

    def summarize(self):
        self._summarize_all_remaining_pairs()

    def get_summarized_text(self):

        return self._summarized_text

class SummarizerManager:

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
