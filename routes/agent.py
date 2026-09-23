"""Agent 模式路由：接收用户消息，返回最终回答与工具调用步骤。"""
from flask import Blueprint, request, jsonify

from services import agent, db
from utils.helpers import format_response

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


@agent_bp.route("/chat", methods=["POST"])
def agent_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or None
    if not message:
        return jsonify(format_response(success=False, error="消息不能为空")), 400

    result = agent.run(message, session_id=session_id)

    if session_id:
        db.append_message(session_id, "user", message)
        db.append_message(session_id, "assistant", result.get("reply", ""))

    return jsonify(format_response(success=True, data=result))
