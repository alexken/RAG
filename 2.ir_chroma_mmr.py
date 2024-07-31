import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
logging.getLogger().setLevel(logging.ERROR)

embeddings = HuggingFaceEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
db = Chroma(persist_directory="./chroma_db_mmr_1000", embedding_function=embeddings)

queries = [    
                "암흑 물질과 암흑 에너지의 차이점", 
                "법인카드는 몇 매씩 배부하는게 원칙인가?",
                "법인카드 사용 제약 조건",
                "회의록 작성시 주의사항",
                "국외 파견직원의 여비 규정에 대해서 알려줘",
            ]

def format_docs(docs):
    return "\n\n\t📋 ".join([d.page_content for d in docs])

for query in queries:
    embedding_vector = embeddings.embed_query(query)
    # docs = db.similarity_search_by_vector_with_relevance_scores(embedding_vector, 3)
    docs = db.max_marginal_relevance_search_by_vector(embedding_vector, k=5, fetch_k=10, lambda_mult=1)
    print(f">>> QUERY: {query} \n>>> RESULT:")
    for d in docs:
        print(f"🗎 {d.page_content}\n  ▫️ 출처: {d.metadata['source']}\n")
    print("─" * 100)