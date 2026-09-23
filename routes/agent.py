"""Agent 模式路由：普通调用 + SSE 流式调用。"""
import json
from flask import Blueprint, request, Response, jsonify, stream_with_context

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


@agent_bp.route("/chat/stream", methods=["POST"])
def agent_chat_stream():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or None
    if not message:
        return jsonify(format_response(success=False, error="消息不能为空")), 400

    def gen():
        final_reply = ""
        for ev in agent.run_stream(message, session_id=session_id):
            if ev["event"] == "done":
                final_reply = ev["data"].get("reply", "")
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"
        if session_id:
            db.append_message(session_id, "user", message)
            db.append_message(session_id, "assistant", final_reply)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
