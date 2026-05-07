from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langsmith import traceable
from dotenv import load_dotenv
import time

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Knowledge base mẫu
DOCS = [
    "LangSmith là platform trace, debug và evaluate LLM applications.",
    "RAGAS đánh giá RAG với metrics: faithfulness, answer relevancy, context precision.",
    "Prompt A/B testing so sánh 2 prompt trên cùng dataset để chọn cái tốt hơn.",
    "LLM-as-judge dùng LLM mạnh để chấm điểm output của LLM khác tự động.",
    "Helicone là proxy cho LLM APIs giúp cache responses và theo dõi costs.",
    "Semantic caching lưu câu trả lời tương tự để tránh gọi API lặp lại.",
]

documents = [Document(page_content=d) for d in DOCS]
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="quality-demo")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

prompt = ChatPromptTemplate.from_template("""
Trả lời câu hỏi dựa trên context. Chỉ dùng thông tin từ context.
Nếu không có thông tin, nói rõ: "Không có thông tin trong tài liệu."

Context: {context}
Câu hỏi: {question}
Trả lời ngắn gọn:""")

def format_docs(docs):
    return "".join(doc.page_content for doc in docs)

@traceable(name="hybrid-rag-query", tags=["rag", "production"])
def rag_query(question: str) -> dict:
    """RAG query với LangSmith tracing tự động."""
    start = time.time()

    docs = hybrid_retriever.invoke(question)
    context = format_docs(docs)

    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    answer = chain.invoke(question)
    latency = round((time.time() - start) * 1000)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved_docs": [d.page_content for d in docs],
        "latency_ms": latency
    }