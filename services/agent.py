"""Agent 核心：基于 Function Calling 的工具自动编排（ReAct 循环）。"""
import json
import time
from typing import Callable, Dict, List

import requests

from config import config
from services.chart_service import ChartService
from services.nlp_service import NLPService
from utils.helpers import read_text_file_content
from services import db

MAX_ROUNDS = 6
TOOL_TIMEOUT = 30
MAX_RETRY = 2  # 单次模型调用失败后额外重试次数
RETRY_BACKOFF = 1.5  # 每次重试间隔（秒），递增
HISTORY_KEEP = 10          # 最近保留原文的消息条数
HISTORY_SUMMARY_THRESHOLD = 20  # 超过此条数才触发摘要压缩
CONFIRM_TOOLS = {"export_report"}  # 需要用户确认才执行的工具


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
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "按关键词搜索过去的对话记录，用于回忆之前聊过什么。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时信息，如新闻、天气、最新数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索词"}
                },
                "required": ["query"],
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


def _impl_search_history(keyword: str) -> str:
    rows = db.search_messages(keyword)
    if not rows:
        return f"没有找到包含「{keyword}」的历史对话。"
    return json.dumps(rows, ensure_ascii=False)


def _impl_web_search(query: str) -> str:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=6,
        )
        data = r.json()
        ans = data.get("AbstractText") or data.get("Answer") or ""
        related = [t.get("Text") for t in data.get("RelatedTopics", []) if t.get("Text")][:3]
        parts = [ans] + related
        parts = [x for x in parts if x]
        return "联网结果：" + ("\n".join(parts) if parts else "未找到直接结果，建议换个关键词。")
    except Exception:
        return "联网搜索暂不可用（当前网络无法访问搜索引擎），请改用已有工具或稍后再试。"


_TOOL_IMPLS: Dict[str, Callable[[Dict], str]] = {
    "read_file": lambda a: _impl_read_file(a["path"]),
    "analyze_text": lambda a: _impl_analyze_text(a["text"]),
    "draw_chart": lambda a: _impl_draw_chart(a["file_path"]),
    "calculate": lambda a: _impl_calculate(a["expression"]),
    "export_report": lambda a: _impl_export_report(a["title"], a["content"]),
    "describe_data": lambda a: _impl_describe_data(a["file_path"]),
    "search_history": lambda a: _impl_search_history(a["keyword"]),
    "web_search": lambda a: _impl_web_search(a["query"]),
}


def _compress_history(history: List[Dict]) -> List[Dict]:
    """历史过长时，把老消息总结成一条 system 摘要，保留最近原文。"""
    if len(history) <= HISTORY_SUMMARY_THRESHOLD:
        return history[-HISTORY_KEEP:]
    old = history[:-HISTORY_KEEP]
    keep = history[-HISTORY_KEEP:]
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in old)
    try:
        data = _call_llm([
            {"role": "system", "content": "用简洁中文总结以下对话历史，保留关键事实、决策和用户偏好，不超过150字。"},
            {"role": "user", "content": convo},
        ])
        summary = data["choices"][0]["message"]["content"]
        return [{"role": "system", "content": f"[历史摘要] {summary}"}] + keep
    except Exception:
        return keep


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


def _call_llm_stream(messages: List[str]):
    """以 stream=True 调用模型，逐块产出内容文本。"""
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
        "stream": True,
    }
    resp = requests.post(url, headers=headers, json=payload,
                         timeout=config.CHAT_TIMEOUT, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        delta = chunk["choices"][0].get("delta", {})
        # 流式下也要累积可能的 tool_calls
        for tc in (delta.get("tool_calls") or []):
            yield {"type": "tool_call_delta", "data": tc}
        text = delta.get("content")
        if text:
            yield {"type": "token", "data": text}


def run(user_message: str, session_id: str = None) -> Dict:
    """跑一轮 Agent：模型自主调用工具，返回最终回答与思考步骤。"""
    history = db.get_history(session_id) if session_id else []
    messages = _compress_history(history)
    messages.append({"role": "user", "content": user_message})
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
            try:
                observation = _TOOL_IMPLS[name](args)
            except Exception as e:  # 工具失败不中断 Agent，反馈给模型
                observation = f"工具执行失败：{e}"
            steps.append({"tool": name, "args": args, "result": observation[:300]})
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation,
            })

    return {"reply": "达到最大工具调用轮数，已停止。", "steps": steps}


def run_stream(user_message: str, session_id: str = None):
    """流式跑 Agent：逐事件产出 {event, data}，供 SSE 推送。"""
    history = db.get_history(session_id) if session_id else []
    messages = _compress_history(history)
    messages.append({"role": "user", "content": user_message})
    steps: List[Dict] = []

    for _ in range(MAX_ROUNDS):
        yield {"event": "thinking", "data": "思考中..."}
        try:
            data = _call_llm(messages)
        except requests.RequestException as e:
            yield {"event": "error", "data": f"调用大模型失败：{e}"}
            return
        message = data["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # 真逐 token 流式：重新以 stream=True 调用，逐块推送
            full = ""
            try:
                for ev in _call_llm_stream(messages):
                    if ev["type"] == "token":
                        full += ev["data"]
                        yield {"event": "token", "data": ev["data"]}
            except requests.RequestException as e:
                yield {"event": "error", "data": f"流式调用失败：{e}"}
                return
            yield {"event": "done", "data": {"reply": full, "steps": steps}}
            return

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"event": "tool_start", "data": {"tool": name, "args": args}}
            if name in CONFIRM_TOOLS:
                yield {"event": "need_confirm", "data": {"tool": name, "args": args}}
                observation = "该操作已暂停，等待用户在前端确认后执行。"
            else:
                try:
                    observation = _TOOL_IMPLS[name](args)
                except Exception as e:
                    observation = f"工具执行失败：{e}"
            steps.append({"tool": name, "args": args, "result": observation[:300]})
            yield {"event": "tool_end", "data": {"tool": name, "result": observation[:300]}}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": observation,
            })

    yield {"event": "done", "data": {"reply": "达到最大工具调用轮数，已停止。", "steps": steps}}
