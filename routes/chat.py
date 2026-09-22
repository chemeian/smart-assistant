"""
Chat API route - handles conversation with LLM providers.
Supports single/multi-turn, image upload, file upload, and session management.
"""
import os
from datetime import datetime
from flask import Blueprint, request, jsonify

from services.llm_service import LLMService
from utils.helpers import (
    format_response, generate_session_id,
    allowed_chat_file, image_to_base64, read_text_file_content, save_uploaded_file
)
from config import config

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")

# In-memory conversation history (per session_id)
_conversations = {}

# Session metadata for conversation list
_sessions = {}


def _trim_history(history, limit):
    """Trim conversation history to stay within limit pairs."""
    if len(history) > limit * 2 + 2:
        history[:] = history[:2] + history[-(limit * 2):]


def _get_session_title(history):
    """Extract a title from a conversation history."""
    for msg in history:
        if msg.get("role") == "user":
            text = msg.get("content", "").strip()
            if text:
                return (text[:30] + "...") if len(text) > 30 else text
    return "新对话"


@chat_bp.route("/send", methods=["POST"])
def send_message():
    """Send a message to the LLM and get a response.
    Accepts JSON body: {message, provider, session_id, mode, image_uri, temperature}
    Mode: 'single' (no history) or 'multi' (with history, default).
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    provider = data.get("provider", "deepseek")
    session_id = data.get("session_id", request.remote_addr or generate_session_id())
    mode = data.get("mode", "multi")
    image_uri = data.get("image_uri", "")
    temperature = float(data.get("temperature", 0.7))

    if not message and not image_uri:
        return jsonify(format_response(success=False, error="消息不能为空")), 400
    if not message and image_uri:
        message = "请描述或分析这张图片。"

    try:
        llm = LLMService(provider)
    except ValueError as e:
        return jsonify(format_response(success=False, error=str(e))), 400

    # Prepare message list
    if mode == "single":
        messages = [{"role": "user", "content": message}]
    else:
        if session_id not in _conversations:
            _conversations[session_id] = []
            # Record session metadata
            _sessions[session_id] = {
                "session_id": session_id,
                "title": _get_session_title([{"role": "user", "content": message}]),
                "first_message": message,
                "message_count": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        history = _conversations[session_id]
        history.append({"role": "user", "content": message})
        _trim_history(history, config.CHAT_HISTORY_LIMIT)
        messages = list(history)

    # Send request (with or without image)
    if image_uri and llm.supports_vision:
        reply = llm.chat_with_image(messages, image_uri, temperature=temperature)
    else:
        reply = llm.chat(messages, temperature=temperature)

    # Update history for multi-turn mode
    if mode == "multi":
        history = _conversations[session_id]
        history.append({"role": reply["role"], "content": reply["content"]})
        if session_id in _sessions:
            _sessions[session_id]["message_count"] = len(history)
            _sessions[session_id]["updated_at"] = datetime.now().isoformat()

    return jsonify(format_response(success=True, data={
        "reply": reply["content"],
        "provider": reply.get("provider", provider),
        "model": reply.get("model", ""),
        "usage": reply.get("usage", {}),
        "session_id": session_id,
        "mode": mode,
    }))


@chat_bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload an image or text file for chat context."""
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

    upload_dir = config.CHAT_UPLOAD_FOLDER
    path = save_uploaded_file(file, upload_dir)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext in config.CHAT_ALLOWED_IMAGE_EXT:
        img_uri = image_to_base64(path)
        if not img_uri:
            return jsonify(format_response(success=False, error="图片处理失败")), 500
        return jsonify(format_response(success=True, data={
            "type": "image",
            "filename": os.path.basename(path),
            "image_uri": img_uri,
            "mime_type": f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}",
            "size": os.path.getsize(path),
        }))

    elif ext in config.CHAT_ALLOWED_TEXT_EXT:
        content = read_text_file_content(path)
        preview = content[:500] if content else ""
        return jsonify(format_response(success=True, data={
            "type": "text",
            "filename": os.path.basename(path),
            "content": content,
            "file_path": path,
            "preview": preview,
            "size": os.path.getsize(path),
        }))

    return jsonify(format_response(success=False, error="不支持的文件类型")), 400


@chat_bp.route("/history", methods=["GET"])
def get_history():
    """Get conversation history for a session."""
    session_id = request.args.get("session_id", request.remote_addr or "")
    history = _conversations.get(session_id, [])
    return jsonify(format_response(success=True, data={
        "session_id": session_id,
        "messages": history,
        "count": len(history),
    }))


@chat_bp.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history for a session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", request.remote_addr or "")
    _conversations.pop(session_id, None)
    _sessions.pop(session_id, None)
    return jsonify(format_response(success=True, message="对话历史已清除"))


@chat_bp.route("/providers", methods=["GET"])
def list_providers():
    """Show available LLM providers and their status."""
    models = LLMService.list_models()
    return jsonify(format_response(success=True, data={"providers": models}))


# --- Session Management ---
@chat_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all saved conversation sessions, ordered by most recent."""
    session_list = sorted(
        _sessions.values(),
        key=lambda s: s.get("updated_at", ""),
        reverse=True
    )
    return jsonify(format_response(success=True, data={"sessions": session_list}))


@chat_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a specific session and its history."""
    if session_id in _conversations:
        del _conversations[session_id]
    if session_id in _sessions:
        del _sessions[session_id]
        return jsonify(format_response(success=True, message="对话已删除"))
    return jsonify(format_response(success=False, error="会话不存在")), 404
