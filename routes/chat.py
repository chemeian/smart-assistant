"""对话相关路由：只负责接参、调 service、拼响应。"""
import os
from flask import Blueprint, request, jsonify

from services import chat_service, db
from services.llm_service import LLMService
from utils.helpers import (
    format_response, generate_session_id,
    allowed_chat_file, image_to_base64, read_text_file_content, save_uploaded_file,
)
from config import config

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@chat_bp.route("/send", methods=["POST"])
def send_message():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    provider = data.get("provider", "deepseek")
    session_id = data.get("session_id") or request.remote_addr or generate_session_id()
    mode = data.get("mode", "multi")
    image_uri = data.get("image_uri", "")
    temperature = float(data.get("temperature", 0.7))

    try:
        result = chat_service.send(
            message, provider, session_id, mode, image_uri, temperature)
    except ValueError as e:
        return jsonify(format_response(success=False, error=str(e))), 400

    result["session_id"] = session_id
    result["mode"] = mode
    return jsonify(format_response(success=True, data=result))


@chat_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify(format_response(success=False, error="未上传文件")), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify(format_response(success=False, error="文件名为空")), 400
    if not allowed_chat_file(file.filename, for_images=False):
        return jsonify(format_response(
            success=False,
            error=f"不支持的文件类型。图片: {', '.join(config.CHAT_ALLOWED_IMAGE_EXT)} | "
                  f"文本: {', '.join(config.CHAT_ALLOWED_TEXT_EXT)}"
        )), 400

    max_bytes = config.CHAT_MAX_FILE_SIZE_MB * 1024 * 1024
    file.stream.seek(0, os.SEEK_END)
    fsize = file.stream.tell()
    file.stream.seek(0)
    if fsize > max_bytes:
        return jsonify(format_response(
            success=False,
            error=f"文件超过 {config.CHAT_MAX_FILE_SIZE_MB}MB 限制（当前 {fsize//1024//1024}MB）"
        )), 400

    path = save_uploaded_file(file, config.CHAT_UPLOAD_FOLDER)
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext in config.CHAT_ALLOWED_IMAGE_EXT:
        try:
            from PIL import Image
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            return jsonify(format_response(success=False, error="图片无法解析，请确认是有效图片")), 400
        img_uri = image_to_base64(path)
        if not img_uri:
            return jsonify(format_response(success=False, error="图片处理失败")), 500
        return jsonify(format_response(success=True, data={
            "type": "image",
            "filename": os.path.basename(path),
            "image_uri": img_uri,
            "mime_type": f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}",
            "size": os.path.getsize(path),
            "width": width,
            "height": height,
        }))

    if ext in config.CHAT_ALLOWED_TEXT_EXT:
        content = read_text_file_content(path)
        return jsonify(format_response(success=True, data={
            "type": "text",
            "filename": os.path.basename(path),
            "content": content,
            "file_path": path,
            "preview": (content or "")[:500],
            "size": os.path.getsize(path),
        }))

    return jsonify(format_response(success=False, error="不支持的文件类型")), 400


@chat_bp.route("/history", methods=["GET"])
def get_history():
    session_id = request.args.get("session_id") or request.remote_addr or ""
    history = db.get_history(session_id)
    return jsonify(format_response(success=True, data={
        "session_id": session_id,
        "messages": history,
        "count": len(history),
    }))


@chat_bp.route("/clear", methods=["POST"])
def clear_history():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.remote_addr or ""
    db.clear_messages(session_id)
    return jsonify(format_response(success=True, message="对话历史已清除"))


@chat_bp.route("/providers", methods=["GET"])
def list_providers():
    return jsonify(format_response(success=True, data={
        "providers": LLMService.list_models()}))


@chat_bp.route("/sessions", methods=["GET"])
def list_sessions():
    return jsonify(format_response(success=True, data={
        "sessions": db.list_sessions()}))


@chat_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    if not db.get_history(session_id) and not any(
            s["session_id"] == session_id for s in db.list_sessions()):
        return jsonify(format_response(success=False, error="会话不存在")), 404
    db.delete_session(session_id)
    return jsonify(format_response(success=True, message="对话已删除"))
