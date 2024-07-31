import logging
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
logging.getLogger().setLevel(logging.ERROR)

template =  """Answer the question based only on the following context.
            If you don't know the answer just say you don't know, don't make it up.
            You must generate an answer in Korean: {context}
            Question: {question}
            """
prompt = ChatPromptTemplate.from_template(template) 

embeddings = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
db1 = Chroma(persist_directory="./chroma_db_mmr_1000", embedding_function=embeddings)
db2 = Chroma(persist_directory="./chroma_db_cosine", embedding_function=embeddings)
retriever1 = db1.as_retriever(search_type="mmr", search_kwargs={'k':3,'fetch_k':10, 'lambda_mult': 0.7})
retriever2 = db2.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.3})
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
)

model = ChatOllama(model="eeve" )

def format_docs(docs):
    return "\n\n◽ ".join([d.page_content for d in docs])

chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

queries = [    
                "암흑 물질과 암흑 에너지의 차이점", 
                "법인카드는 몇 매씩 배부하는게 원칙인가?",
                "법인카드 사용 제약 조건",
                "회의록 작성시 주의사항",
                "국외 파견직원의 여비 규정에 대해서 알려줘",
           ]

for query in queries:
    print("─" * 100)
    print(">>> QUERY: " + query)
    print(">>> RESULT: \n\t" + "\n\t".join(chain.invoke(query).split("\n")) + "\n\t")