"""
Python Multi-function Smart Assistant - Main application factory.

Flask backend with LLM chat (DeepSeek / Tongyi Qianwen),
and NLP processing capabilities.
"""
import os
import time
import json as _json

import numpy as _np
from flask import Flask, jsonify, render_template, request, redirect
from flask_cors import CORS

from config import config
from routes.chat import chat_bp
from routes.chart import chart_bp
from routes.nlp import nlp_bp
from services import db as _db
from utils.helpers import format_response


class NumpyJSONEncoder(_json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


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

    # --- 接口调用统计埋点 ---
    @app.before_request
    def _stats_start():
        request._t0 = time.time()

    @app.after_request
    def _stats_end(resp):
        try:
            path = request.path
            if path.startswith("/api/") and "/stats" not in path:
                dur = int((time.time() - request._t0) * 1000)
                _db.record_call(path, 200 <= resp.status_code < 400, dur)
        except Exception:
            app.logger.warning("stats record failed", exc_info=True)
        return resp

    @app.route("/api/stats/overview", methods=["GET"])
    def stats_overview():
        return jsonify(format_response(success=True, data=_db.stats_overview()))

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
        })

    @app.route("/api/config", methods=["GET"])
    def get_config():
        safe = {k: v for k, v in config.to_dict().items()
                if "KEY" not in k and "SECRET" not in k}
        safe["deepseek_configured"] = bool(config.DEEPSEEK_API_KEY)
        safe["qwen_configured"] = bool(config.QWEN_API_KEY)
        return jsonify(safe)

    os.makedirs(config.CHAT_UPLOAD_FOLDER, exist_ok=True)
    from services import db as _db
    _db.init_db()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
