import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from datetime import datetime, timezone
import os
logging.getLogger().setLevel(logging.ERROR)

os.system('rm -rf chroma_db_mmr_1000')

t1 = datetime.now(timezone.utc)

raw_documents = DirectoryLoader(path='./docs', glob='*.txt').load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 100)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db_mmr_1000")

t2 = datetime.now(timezone.utc)

print(t2-t1)