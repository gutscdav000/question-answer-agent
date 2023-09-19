from OCRConverter import read_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
#import openai

###################################
#         Questions               #
# - text/docs splitting strategy  #
# - text/docs embedding strategy  #
# NEXT                            #
# - how to use openAI embeddings  #
# - how to cache openAI embeddings#
###################################

def split_docs(documents,chunk_size=1000,chunk_overlap=250):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.create_documents(documents)
  return docs


text = read_file('the-fast-and-the-furious-2001/rev2_manual_the-fast-and-the-furious-2001')
docs = split_docs([text])
model_id = "text-embedding-ada-002"
open_ai_key = os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
# caching
fs = LocalFileStore("./cache/embeddings/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, fs, namespace=embeddings.model
)

db = Chroma.from_documents(documents = docs,
                           embedding = cached_embedder,
                           collection_name = 'fast-and-furious',
                           persist_directory = './cache/chroma/',
                           )
query = "what happens in the first scene when Brian meets Dom?"
# matching_docs = db.similarity_search(query)
# for d in matching_docs:
#     print("*" * 30)
#     print(d.page_content.strip())


#llm = ChatOpenAI(temperature=0, openai_api_key=open_ai_key)
#qa = ConversationalRetrievalChain.from_llm(promptedLLM, db.as_retriever())
#qa({"question": user_message, "chat_history": history})
#chat_history = []
#response = qa({'question': query, 'chat_history': chat_history})

prompt_template = '''
Given the following extracted parts of the movie script from the fast and the furious and a question regarding the movie, answer the question based on the context you have received.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

CONTEXT: {context}
QUESTION: {question}
=========
Content: ...
Source: ...
...
=========
FINAL ANSWER:
SOURCES:
'''
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
prompt = PromptTemplate(
    input_variables=["question", "context"], template=prompt_template
)
llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=open_ai_key, temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")
chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs=chain_type_kwargs
)

response = qa.run(query)
print(response)

    
