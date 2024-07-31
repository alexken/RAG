import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
logging.getLogger().setLevel(logging.ERROR)

raw_documents = DirectoryLoader(path='./docs', glob='*.txt').load()
text_splitter = CharacterTextSplitter(chunk_size = 100, chunk_overlap  = 20)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db_cosine", collection_metadata={"hnsw:space": "cosine"})
