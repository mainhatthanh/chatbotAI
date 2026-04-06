from flask import Flask, render_template, request, jsonify
from rag.pipeline import RAGPipeline

app = Flask(__name__)

pipeline = RAGPipeline("data/books.csv", "data/faq.json")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"response": "Vui lòng nhập câu hỏi."})

    response, retrieved_docs = pipeline.answer(query, top_k=3)

    return jsonify({
        "response": response,
        "retrieved_docs": retrieved_docs
    })


if __name__ == "__main__":
    app.run(debug=True)