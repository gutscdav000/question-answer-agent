from dotenv import load_dotenv
from langchain.agents.react.base import DocstoreExplorer
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings, SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore, LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore
from OCRConverter import read_file
import os


load_dotenv()

def split_docs(documents,chunk_size=1000,chunk_overlap=250):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.create_documents(documents)
  return docs


text = read_file('the-fast-and-the-furious-2001/rev2_manual_the-fast-and-the-furious-2001')
docs = split_docs([text])
model_id = "text-embedding-ada-002"
open_ai_key = os.getenv('OPENAI_API_KEY')
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

def get_document_from_chroma(query: str) -> Document:
    return db.similarity_search(query)[0]

# initialize docstore
from langchain.docstore import  DocstoreFn
docstoreExplorer = DocstoreExplorer(DocstoreFn(get_document_from_chroma))

tools = [
    Tool(
        name="Search",
        func=docstoreExplorer.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstoreExplorer.lookup,
        description="useful for when you need to ask with lookup",
    ),
]
llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=open_ai_key, temperature=0)
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
query = "what happens in the first scene when Brian meets Dom?"
react.run(query)

# prompt_template = '''
# Given the following extracted parts of the movie script from the fast and the furious and a question regarding the movie, answer the question based on the context you have received.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.

# CONTEXT: {context}
# QUESTION: {question}
# =========
# Content: ...
# Source: ...
# ...
# =========
# FINAL ANSWER:
# SOURCES:
# '''
# document_prompt = PromptTemplate(
#     input_variables=["page_content"],
#     template="{page_content}"
# )
# prompt = PromptTemplate(
#     input_variables=["question", "context"], template=prompt_template
# )
# llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=open_ai_key, temperature=0)
# qa_chain = load_qa_chain(llm, chain_type="stuff")
# chain_type_kwargs = {"prompt": prompt}
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(),
#     chain_type_kwargs=chain_type_kwargs
# )

# query = "what happens in the first scene when Brian meets Dom?"
# response = qa.run(query)
# print(response)

    
