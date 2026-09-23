"""Agent 核心：基于 Function Calling 的工具自动编排（ReAct 循环）。"""
import json
from typing import Callable, Dict, List

import requests

from config import config
from services.chart_service import ChartService
from services.nlp_service import NLPService
from utils.helpers import read_text_file_content

MAX_ROUNDS = 6
TOOL_TIMEOUT = 30


# ---------- 工具定义（写给模型看的说明书） ----------
TOOLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件的文本内容，支持 csv/txt/md/json/docx/pdf 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件绝对路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_text",
            "description": "对一段文本做情感分析、关键词提取和摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待分析的文本"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_chart",
            "description": "对数据文件生成统计图（直方图/柱状图/热力图）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "数据文件路径（csv/xlsx）"}
                },
                "required": ["file_path"],
            },
        },
    },
]


# ---------- 工具实现 ----------
def _impl_read_file(path: str) -> str:
    content = read_text_file_content(path)
    if content is None:
        return "读取失败：文件不存在或无法解析。"
    return content[:4000]


def _impl_analyze_text(text: str) -> str:
    svc = NLPService()
    result = {
        "情感": svc.sentiment_analysis(text).get("data"),
        "关键词": svc.keyword_extraction(text).get("data"),
        "摘要": svc.text_summary(text).get("data"),
    }
    return json.dumps(result, ensure_ascii=False)


def _impl_draw_chart(file_path: str) -> str:
    svc = ChartService()
    result = svc.generate_charts(file_path)
    charts = result.get("charts", []) if result.get("success") else []
    summary = [{"类型": c.get("类型") or c.get("type")} for c in charts]
    return json.dumps({"已生成图表": summary}, ensure_ascii=False)


_TOOL_IMPLS: Dict[str, Callable[[Dict], str]] = {
    "read_file": lambda a: _impl_read_file(a["path"]),
    "analyze_text": lambda a: _impl_analyze_text(a["text"]),
    "draw_chart": lambda a: _impl_draw_chart(a["file_path"]),
}


def _call_deepseek(messages: List[Dict]) -> Dict:
    """调用 DeepSeek，携带 tools 参数。"""
    url = f"{config.DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.5,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=config.CHAT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def run(user_message: str) -> Dict:
    """跑一轮 Agent：模型自主调用工具，返回最终回答与思考步骤。"""
    messages: List[Dict] = [
        {"role": "user", "content": user_message}
    ]
    steps: List[Dict] = []

    for _ in range(MAX_ROUNDS):
        try:
            data = _call_deepseek(messages)
        except requests.RequestException as e:
            return {"reply": f"调用大模型失败：{e}", "steps": steps}
        message = data["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {"reply": message.get("content", ""), "steps": steps}

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            steps.append({"tool": name, "args": args})
            try:
                observation = _TOOL_IMPLS[name](args)
            except Exception as e:  # 工具失败不中断 Agent，反馈给模型
                observation = f"工具执行失败：{e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation,
            })

    return {"reply": "达到最大工具调用轮数，已停止。", "steps": steps}
