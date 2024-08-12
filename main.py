# %% [md]
# # Summarization of Large Pieces of Text with LangChain 
# ## Context
# Modern NLP systems excel at text summarization--extraction of features
# which retain the meaning of the piece of text, as well as reformulating
# them into *new* sentences. However, as the task grows in complexity
# because of inherent limitations of the LLMs to consume, and timely
# process the text, it is necessary to create a specific strategy to 
# perform summarization of larger pieces of text such as whole books.
# ## Objective
# As partially revealed in previous paragraph the objective of this demo
# is to create a summary of the text that is several times greater
# than the current limitation in terms of tokens that can be consumed
# by LLMs. While, the only limitation appears to be number of tokens,
# other limitations need to be taken into account as well: cost, speed,
# overall quality in order to meet the objective.
# ## Technological Stack
# The back-end of the whole system is the LangChain library. However,
# its scope will be limited to utilzation of only some classes, such as:
# `RecursiveCharacterSplitter`, `PromptTemplate`, and `Document`. While,
# `LangChain` does provides classes to perform summarization of large
# pieces of text into chains, there is always a benefit in balancing
# between the utilization of pre-existing classes and custom 
# implementation. In this case custom implementation will handle the 
# splitting of the text and further summarization chain. The choice of
# the LLM has fallen to `OpenAI`s GPT-3.5-Turbo which can consume 4096.
# tokens. The compete system is monitored via `LangSmith` platform.
# ## Methodology
# In order to achieve the objective a following methodology has been
# selected:
#   1. A selection of large pieces of text will be made on the basis of
#   publicly available books in txt format from the https://www.gutenberg.org/.
#   2. Data from these txt files will be loaded as is, and split into
#   sections. The approach for splitting is that regardless of the
#   initial text size, text will always be split to no more than 128
#   sections, which will be of the same size within the document, and
#   can be of different size across different documents.
#   3. Summarization of the sections will be done by aggregating two
#   consecutive pieces-of-text in a binary fashion until all pairs are
#   exhausted.
#   4. Final summary will be an agglomeration of all previous pairs, and
#   within formation of intermediate summaries restrictions will be
#   imposed in terms of number of paragraphs and length of each 
#   paragraph expressed in total number of characters including the
#   whitespaces. Hence, the explanation for not completely relying on
#   `LangChain` classes.
# ## Architecture
# Having in mind that complexity of the task lies in the whole system
# design, the architecutre of the LLM sub-system is fairly simple. It
# is a LLM based on a custom prompt containing instructions on how to
# summarize two pieces of text.
# ## Impementation
# Firstly, let us make the necessary imports.
# %%
import textwrap
from utilities.text import TextLoaderManager
from summarizer.summarizer import TextSplitterManager, SummarizationManager
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
# %% [md]
# Secondly, let us delare a file handle for the folder with the data.
# %%
TEXT_FOLDER = r'C:\Users\marko\Desktop\data'
# %% [md]
# Now let us utilize the `TextLoaderManager` class to load the text 
# files.
# %%
text_loader = TextLoaderManager(folder=TEXT_FOLDER)
text_loader.read_files()
texts = text_loader.get_pieces_of_text()
# %% [md]
# The next step is to split the documents, according to the 
# methodological specification.
# %%
text_splitter = TextSplitterManager(pieces_of_text=texts)
text_splitter.split_documents()
split_documents = text_splitter.get_split_documents()
# %% [md]
# Finally it is possible to start the `SummarizationManager` and to
# create summaries of documents.
# %%
summarizer = SummarizationManager(
    chuncked_pieces_of_text=[split_documents[6]]
)
summarizer.summarize_text()
summarized_text = summarizer.get_summarized_texts()

# %% [md]
# %%
paragraphs = summarized_text[0][0].page_content.split('\n\n')
print(len(paragraphs))

text_wrapper = textwrap.TextWrapper(width=70, fix_sentence_endings=True)

wrapped_paragraphs = []
for paragraph in paragraphs:
    wrapped = text_wrapper.wrap(text=paragraph)
    joined = '\n'.join(wrapped)
    wrapped_paragraphs.extend([joined])

final = '\n\n'.join(wrapped_paragraphs)
figure = plt.figure(figsize=[8.3, 11.7])
plotting_grid = grid.GridSpec(nrows=1, ncols=1)
axis_1 = figure.add_subplot(plotting_grid[0, 0])
text_kwargs = dict(ha='left', va='top', fontsize=10, family='monospace')
axis_1.text(x=.05, y=.95, s=final, wrap=True, **text_kwargs)  
axis_1.set_xticklabels([])
axis_1.set_yticklabels([])
axis_1.tick_params(
    axis='both', 
    which='both', 
    bottom=False, 
    top=False, 
    left=False
)
plt.show()
plt.close()


