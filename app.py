"""
Python Multi-function Smart Assistant - Main application factory.

Flask backend with LLM chat (DeepSeek / Tongyi Qianwen),
and NLP processing capabilities.
"""
import os
from flask import Flask, jsonify, send_from_directory
from flask import render_template

import json as _json
import numpy as _np

class NumpyJSONEncoder(_json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        elif isinstance(obj, (_np.floating,)):
            return float(obj)
        elif isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return float(obj.item()) if isinstance(obj.item(), float) else int(obj.item())
        return super().default(obj)
from flask_cors import CORS

from config import config
from routes.chat import chat_bp
from routes.chart import chart_bp
from routes.nlp import nlp_bp


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.json_encoder = NumpyJSONEncoder
    app.secret_key = config.SECRET_KEY
    app.debug = config.DEBUG

    # Set max upload size
    app.config["MAX_CONTENT_LENGTH"] = config.CHAT_MAX_FILE_SIZE_MB * 1024 * 1024

    # --- CORS ---
    origins = config.CORS_ORIGINS.split(",") if config.CORS_ORIGINS != "*" else "*"
    CORS(app, origins=origins, supports_credentials=True)

    # --- Register blueprints ---
    app.register_blueprint(chat_bp)
    app.register_blueprint(chart_bp)
    app.register_blueprint(nlp_bp)

    # --- Health check ---
    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "app": "Python Multi-function Smart Assistant",
            "version": "1.1.0",
            "providers": {
                "deepseek": bool(config.DEEPSEEK_API_KEY),
                "qwen": bool(config.QWEN_API_KEY),
            }
        })

    # --- Root welcome ---
    @app.route("/", methods=["GET"])
    def index():
        # Force fresh load - redirect with timestamp
        import time
        from flask import request, redirect
        if request.args.get("_t") is None:
            return redirect(f"/?_t={int(time.time())}")
        template_path = os.path.join(app.root_path, "templates", "index.html")
        if os.path.exists(template_path):
            resp = app.make_response(render_template("index.html"))
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp
        return jsonify({
            "app": "Python Multi-function Smart Assistant",
            "version": "1.1.0",
            "endpoints": {
                "chat": {
                    "send": "POST /api/chat/send",
                    "upload": "POST /api/chat/upload",
                    "history": "GET /api/chat/history",
                    "clear": "POST /api/chat/clear",
                    "providers": "GET /api/chat/providers",
                },
            "analysis": {
                "upload": "POST /api/analysis/upload",
                "describe": "POST /api/analysis/describe",
                "chart": "POST /api/analysis/chart",
                "correlation": "POST /api/analysis/correlation",
                "files": "GET /api/analysis/reports",
                },
                "chart": {
                    "generate": "POST /api/chart/generate",
                    "stats": "POST /api/chart/stats",
                },
                "nlp": {
                    "sentiment": "POST /api/nlp/sentiment",
                    "keywords": "POST /api/nlp/keywords",
                    "summary": "POST /api/nlp/summary",
                    "word_count": "POST /api/nlp/word-count",
                    "classify": "POST /api/nlp/classify",
                    "entities": "POST /api/nlp/entities",
                    "readability": "POST /api/nlp/readability",
                    "full_analyze": "POST /api/nlp/analyze",
                },
                "system": {
                    "health": "GET /api/health",
                    "config": "GET /api/config",
                },
            }
        })

    @app.route("/api/config", methods=["GET"])
    def get_config():
        safe = {k: v for k, v in config.to_dict().items()
                if "KEY" not in k and "SECRET" not in k}
        safe["deepseek_configured"] = bool(config.DEEPSEEK_API_KEY)
        safe["qwen_configured"] = bool(config.QWEN_API_KEY)
        return jsonify(safe)

    os.makedirs(config.CHAT_UPLOAD_FOLDER, exist_ok=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
