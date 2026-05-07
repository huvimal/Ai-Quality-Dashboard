from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json, re

load_dotenv()
judge_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

def llm_judge(question: str, answer: str, context: str) -> dict:
    """Chấm điểm câu trả lời từ 1-10 theo 3 tiêu chí."""
    prompt = f"""Chấm điểm câu trả lời RAG này. Trả về JSON.

Câu hỏi: {question}
Context cung cấp: {context[:500]}
Câu trả lời: {answer}

Chấm điểm từ 1-10 theo 3 tiêu chí:
- faithfulness: câu trả lời có dựa trên context không?
- relevancy: câu trả lời có liên quan đến câu hỏi không?  
- conciseness: có ngắn gọn, không dài dòng không?

Chỉ trả về JSON:
{{"faithfulness": 8, "relevancy": 9, "conciseness": 7, "overall": 8, "feedback": "nhận xét ngắn"}}"""

    r = judge_llm.invoke(prompt)
    try:
        match = re.search(r'{.*}', r.content, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except:
        return {"faithfulness": 0, "relevancy": 0, "conciseness": 0, "overall": 0, "feedback": "error"}

def evaluate_response(query_result: dict) -> dict:
    """Đánh giá một query result, trả về scores."""
    scores = llm_judge(
        question=query_result["question"],
        answer=query_result["answer"],
        context=query_result["context"]
    )
    return {
        **query_result,
        "scores": scores,
        "quality_label": (
            "Tốt" if scores.get("overall", 0) >= 7 else
            "Trung bình" if scores.get("overall", 0) >= 5 else
            "Cần cải thiện"
        )
    }