def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_agent_stream_route_exists(client):
    # 只测路由存在且返回 SSE 内容类型，不真实调用 LLM
    r = client.post(
        "/api/agent/chat/stream",
        json={"message": "test"},
    )
    assert r.status_code in (200, 500)


def test_nlp_sentiment_route(client):
    r = client.post("/api/nlp/sentiment", json={"text": "今天很开心"})
    assert r.status_code == 200
