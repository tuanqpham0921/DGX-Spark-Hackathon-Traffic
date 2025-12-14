import os, flask, requests
from flask import request, jsonify

app = flask.Flask(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("messages")
    if not msg:
        return jsonify({"error": "Missing 'messages' field"}), 400

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": msg,
            "stream": False
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        # Ollama non-stream chat returns { message: {...}, done: true, ... }
        answer = j.get("message", {}).get("content", "")
        return jsonify({
            "answer": answer,
            "raw": j
        })
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)