def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_agent_stream_route_no_key(client, monkeypatch):
    # 没有真实 key 时应优雅返回错误，而不是 500 崩溃
    monkeypatch.setenv("QWEN_API_KEY", "")
    r = client.post(
        "/api/agent/chat/stream",
        json={"message": "你好"},
    )
    assert r.status_code == 200


def test_nlp_sentiment_route(client):
    r = client.post("/api/nlp/sentiment", json={"text": "今天很开心"})
    assert r.status_code == 200
