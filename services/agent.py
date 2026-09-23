"""Agent 核心：基于 Function Calling 的工具自动编排（ReAct 循环）。"""
import json
import time
from typing import Callable, Dict, List

import requests

from config import config
from services.chart_service import ChartService
from services.nlp_service import NLPService
from utils.helpers import read_text_file_content

MAX_ROUNDS = 6
TOOL_TIMEOUT = 30
MAX_RETRY = 2  # 单次模型调用失败后额外重试次数
RETRY_BACKOFF = 1.5  # 每次重试间隔（秒），递增


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
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "安全计算一个数学表达式，如 128*0.15、(120+80)/2。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，只含数字和+-*/()."}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_report",
            "description": "把分析结果导出为 Markdown 报告文件保存到本地。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "报告标题"},
                    "content": {"type": "string", "description": "报告正文（Markdown）"}
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "读取数据文件（csv/xlsx）并给出行列数、字段名和数值列概况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "数据文件路径"}
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


def _impl_calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return "表达式非法，只支持数字和 + - * / ( )"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


def _impl_export_report(title: str, content: str) -> str:
    import os, re
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fa5-]", "_", title)[:40] or "report"
    path = os.path.join(out_dir, safe + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}\n")
    return f"报告已保存：{path}"


def _impl_describe_data(file_path: str) -> str:
    import pandas as pd
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        info = {
            "行数": int(df.shape[0]),
            "列数": int(df.shape[1]),
            "字段": list(df.columns),
            "数值列均值": df.select_dtypes("number").mean().round(2).to_dict(),
        }
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return f"读取数据失败：{e}"


_TOOL_IMPLS: Dict[str, Callable[[Dict], str]] = {
    "read_file": lambda a: _impl_read_file(a["path"]),
    "analyze_text": lambda a: _impl_analyze_text(a["text"]),
    "draw_chart": lambda a: _impl_draw_chart(a["file_path"]),
    "calculate": lambda a: _impl_calculate(a["expression"]),
    "export_report": lambda a: _impl_export_report(a["title"], a["content"]),
    "describe_data": lambda a: _impl_describe_data(a["file_path"]),
}


def _call_llm(messages: List[Dict]) -> Dict:
    """调用大模型（通义千问，OpenAI 兼容接口），携带 tools 参数。"""
    url = f"{config.QWEN_API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.QWEN_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.5,
    }
    last_error = None
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=config.CHAT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < MAX_RETRY:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise last_error


def run(user_message: str) -> Dict:
    """跑一轮 Agent：模型自主调用工具，返回最终回答与思考步骤。"""
    messages: List[Dict] = [
        {"role": "user", "content": user_message}
    ]
    steps: List[Dict] = []

    for _ in range(MAX_ROUNDS):
        try:
            data = _call_llm(messages)
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
