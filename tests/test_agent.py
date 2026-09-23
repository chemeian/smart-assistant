import os
import pytest

os.environ.setdefault("QWEN_API_KEY", "test")

from services import agent


def test_calculate_basic():
    out = agent._TOOL_IMPLS["calculate"]({"expression": "1+2*3"})
    assert "7" in str(out)


def test_calculate_error():
    out = agent._TOOL_IMPLS["calculate"]({"expression": "1+abc"})
    assert len(str(out)) > 0


def test_analyze_text():
    out = agent._TOOL_IMPLS["analyze_text"]({"text": "这家店服务很差，再也不来了"})
    assert isinstance(out, str) and len(out) > 0


def test_search_history():
    out = agent._TOOL_IMPLS["search_history"]({"keyword": "你好"})
    assert isinstance(out, str)


def test_export_report_in_set():
    assert "export_report" in agent.CONFIRM_TOOLS


def test_tools_registered():
    names = [t["function"]["name"] for t in agent.TOOLS]
    for need in ("calculate", "analyze_text", "export_report", "search_history"):
        assert need in names


def test_history_compression_short():
    msgs = [{"role": "user", "content": "hi"}]
    assert agent._compress_history(msgs) == msgs
