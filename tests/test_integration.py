def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_chat_stream_route(client):
    r = client.post(
        "/api/agent/chat/stream",
        json={"message": "你好", "session_id": None},
    )
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "event:" in body


def test_nlp_sentiment_route(client):
    r = client.post("/api/nlp/sentiment", json={"text": "今天很开心"})
    assert r.status_code == 200
