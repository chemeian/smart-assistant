"""Agent 模式路由：接收用户消息，返回最终回答与工具调用步骤。"""
from flask import Blueprint, request, jsonify

from services import agent
from utils.helpers import format_response

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


@agent_bp.route("/chat", methods=["POST"])
def agent_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(format_response(success=False, error="消息不能为空")), 400

    result = agent.run(message)
    return jsonify(format_response(success=True, data=result))
