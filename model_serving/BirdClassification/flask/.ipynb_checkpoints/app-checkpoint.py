from flask import Flask, jsonify, request, render_template
import requests
import os

app = Flask(__name__)

# === Config ===
FASTAPI_URL = os.getenv("FASTAPI_SERVER_URL", "http://fastapi_server:8000")

@app.route("/")
def home():
    return "Flask is running. Use /ping, /upload (UI), or /predict-audio (API)."

@app.route("/ping")
def ping_fastapi():
    try:
        response = requests.get(f"{FASTAPI_URL}/docs")
        return jsonify({
            "fastapi_url": FASTAPI_URL,
            "status": "reachable",
            "code": response.status_code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === API route for external clients like curl or Python scripts ===
@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            files={"file": (file.filename, file.stream, file.content_type)},
            timeout=30
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Web-based upload form ===
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_files = request.files.getlist("file")
        if not uploaded_files:
            return "No files uploaded", 400

        # Prepare files for FastAPI
        form_data = [
            ("files", (f.filename, f.stream, f.content_type)) for f in uploaded_files
        ]

        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                files=form_data,
                timeout=60
            )
            predictions = response.json()

            # Ensure we always get a list of dicts
            if isinstance(predictions, dict):
                predictions = [predictions]
            elif isinstance(predictions, str):
                predictions = [{"error": predictions}]

            return render_template("result.html", predictions=predictions)

        except Exception as e:
            return render_template("result.html", predictions=[{"error": str(e)}])

    return render_template("upload.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
