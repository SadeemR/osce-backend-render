from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # يسمح للتطبيق بالاتصال من iOS


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    # 🔹 هنا ضع كود استدعاء BioMistral أو الـ API الخارجي
    return jsonify({"reply": f"تم استلام: {message}"})


@app.route("/evaluate", methods=["POST"])
def evaluate():
    # 🔹 هنا ضع كود التقييم
    return jsonify({"empathy_score": 8.0, "clarity_score": 7.5, "total_score": 8.2, "feedback": "أداء جيد"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
