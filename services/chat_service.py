"""对话业务编排：历史管理、模型选择、视觉降级、持久化。"""
from typing import Dict, Optional

from config import config
from services.llm_service import LLMService
from services import db


VISION_PROVIDER = "qwen"


def _trim_history(history, limit):
    """保留首条系统消息 + 最近 limit 轮对话。"""
    if len(history) > limit * 2 + 2:
        del history[2:len(history) - limit * 2]


def _session_title(history) -> str:
    for msg in history:
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if text:
                return text[:30] + ("..." if len(text) > 30 else "")
    return "新对话"


def _build_messages(mode: str, message: str, session_id: str):
    """组装发给大模型的消息列表。"""
    if mode == "single":
        return [{"role": "user", "content": message}]
    history = db.get_history(session_id)
    history.append({"role": "user", "content": message})
    _trim_history(history, config.CHAT_HISTORY_LIMIT)
    return history


def send(message: str, provider: str, session_id: str,
         mode: str, image_uri: str = "",
         temperature: float = 0.7) -> Dict:
    """处理一轮对话，返回回复内容与元信息。"""
    if not message and not image_uri:
        raise ValueError("消息不能为空")
    if not message and image_uri:
        message = "请描述或分析这张图片。"

    llm = LLMService(provider)
    messages = _build_messages(mode, message, session_id)

    # 带图片但当前模型不支持看图时，自动切到视觉模型
    if image_uri and not llm.supports_vision:
        llm = LLMService(VISION_PROVIDER)

    if image_uri and llm.supports_vision:
        reply = llm.chat_with_image(messages, image_uri, temperature=temperature)
    else:
        reply = llm.chat(messages, temperature=temperature)

    if mode == "multi":
        db.append_message(session_id, "user", message)
        db.append_message(session_id, reply["role"], reply["content"])
        db.touch_session(session_id, _session_title(messages))

    return {
        "reply": reply["content"],
        "provider": reply.get("provider", provider),
        "model": reply.get("model", ""),
        "usage": reply.get("usage", {}),
    }
