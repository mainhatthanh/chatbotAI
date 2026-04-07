import os

from flask import Flask, render_template, request, jsonify
from rag.pipeline import RAGPipeline

app = Flask(__name__)

BOOKS_PATH = "data/books.csv"
FAQ_PATH = "data/faq.json"

pipeline = None
pipeline_mtimes = {}


def get_pipeline():
    global pipeline, pipeline_mtimes

    current_mtimes = {
        BOOKS_PATH: os.path.getmtime(BOOKS_PATH),
        FAQ_PATH: os.path.getmtime(FAQ_PATH),
    }

    if pipeline is None or current_mtimes != pipeline_mtimes:
        pipeline = RAGPipeline(BOOKS_PATH, FAQ_PATH)
        pipeline_mtimes = current_mtimes

    return pipeline


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"response": "Vui lòng nhập câu hỏi."})

    response, retrieved_docs = get_pipeline().answer(query, top_k=3)

    return jsonify({
        "response": response,
        "retrieved_docs": retrieved_docs
    })


if __name__ == "__main__":
    app.run(debug=True)
