from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from backend.app.config import Config
from backend.app.guardrails import validate
from backend.app.oracle import OracleError, ask

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.config.from_object(Config)

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.post("/api/oracle")
    def oracle():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "")

        result = validate(question)
        if not result.valid:
            return jsonify(ok=False, error=result.reason), 400

        try:
            answer = ask(question.strip())
        except OracleError as exc:
            return jsonify(ok=False, error=str(exc)), 502

        return jsonify(ok=True, answer=answer)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, port=Config.PORT)
