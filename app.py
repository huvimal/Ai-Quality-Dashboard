import gradio as gr
from rag_pipeline import rag_query
from evaluator import evaluate_response

# Lưu lịch sử queries để hiển thị dashboard
query_history = []

def ask_and_evaluate(question: str):
    """Hỏi RAG, đánh giá, lưu vào history."""
    if not question.strip():
        return "⚠️ Nhập câu hỏi!", "", "", "", ""

    result = rag_query(question)
    evaluated = evaluate_response(result)
    query_history.append(evaluated)

    scores = evaluated.get("scores", {})
    score_display = (
        f"Faithfulness:  {scores.get('faithfulness', 0)}/10"
        f"Relevancy:     {scores.get('relevancy', 0)}/10"
        f"Conciseness:   {scores.get('conciseness', 0)}/10"
        f"Overall:       {scores.get('overall', 0)}/10"
        f"Nhận xét: {scores.get('feedback', '')}"
    )

    return (
        evaluated["answer"],
        evaluated["context"][:400] + "...",
        score_display,
        evaluated["quality_label"],
        f"⏱️ {evaluated['latency_ms']}ms"
    )

def get_dashboard_stats():
    """Tổng hợp thống kê từ tất cả queries."""
    if not query_history:
        return "Chưa có dữ liệu. Hãy thử hỏi vài câu trước!"

    total = len(query_history)
    avg_faith = sum(q.get("scores", {}).get("faithfulness", 0) for q in query_history) / total
    avg_rel   = sum(q.get("scores", {}).get("relevancy", 0) for q in query_history) / total
    avg_over  = sum(q.get("scores", {}).get("overall", 0) for q in query_history) / total
    avg_lat   = sum(q.get("latency_ms", 0) for q in query_history) / total
    good      = sum(1 for q in query_history if q.get("quality_label") == "Tốt")

    return (
        f"📊 DASHBOARD STATS ({total} queries)"
        f"{'='*40}"
        f"Faithfulness trung bình:  {avg_faith:.1f}/10"
        f"Relevancy trung bình:     {avg_rel:.1f}/10"
        f"Overall trung bình:       {avg_over:.1f}/10"
        f"Latency trung bình:       {avg_lat:.0f}ms"
        f"Tỷ lệ 'Tốt':             {good}/{total} ({good/total*100:.0f}%)"
        f"🔗 LangSmith traces: smith.langchain.com"
    )

with gr.Blocks(title="AI Quality Dashboard") as demo:
    gr.Markdown("# 📊 AI Quality Dashboard RAG Pipeline với LangSmith Tracing + LLM-as-judge Evaluation")

    with gr.Tab("🔍 Hỏi & Đánh giá"):
        question_input = gr.Textbox(
            placeholder="Nhập câu hỏi...",
            label="Câu hỏi"
        )
        ask_btn = gr.Button("🔍 Hỏi & Đánh giá", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                answer_out  = gr.Textbox(label="Câu trả lời RAG", lines=4, interactive=False)
                context_out = gr.Textbox(label="Context retrieved", lines=4, interactive=False)
            with gr.Column(scale=1):
                score_out   = gr.Textbox(label="Điểm đánh giá", lines=6, interactive=False)
                quality_out = gr.Textbox(label="Chất lượng", interactive=False)
                latency_out = gr.Textbox(label="Latency", interactive=False)

        gr.Examples(
            examples=[
                ["LangSmith dùng để làm gì?"],
                ["RAGAS đo lường những gì?"],
                ["Helicone giúp tiết kiệm chi phí thế nào?"],
                ["Prompt injection là gì?"],
            ],
            inputs=question_input
        )

        ask_btn.click(
            fn=ask_and_evaluate,
            inputs=[question_input],
            outputs=[answer_out, context_out, score_out, quality_out, latency_out]
        )

    with gr.Tab("📊 Dashboard Stats"):
        refresh_btn = gr.Button("🔄 Cập nhật thống kê", variant="primary")
        stats_out   = gr.Textbox(label="Thống kê tổng hợp", lines=15, interactive=False)
        refresh_btn.click(fn=get_dashboard_stats, outputs=[stats_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)