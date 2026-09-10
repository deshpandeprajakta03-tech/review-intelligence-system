from flask import Flask, request, jsonify, render_template
from app.classifier import predict_pipeline

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def classify_review():
    data = request.json
    review_text = data.get("review", "")
    if not review_text:
        return jsonify({"error": "No review provided"}), 400

    result = predict_pipeline(review_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
