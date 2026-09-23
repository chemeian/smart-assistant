"""共享 pytest fixture：提供 Flask 测试客户端。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c
