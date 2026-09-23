"""接口冒烟测试：健康检查、对话、NLP、上传与图表。"""
import io


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_root_page(client):
    r = client.get("/", follow_redirects=True)
    assert r.status_code == 200


def test_providers(client):
    r = client.get("/api/chat/providers")
    assert r.get_json()["success"]


def test_send_invalid_provider(client):
    r = client.post("/api/chat/send", json={"message": "hi", "provider": "nope"})
    assert r.status_code == 400


def test_send_empty_message(client):
    r = client.post("/api/chat/send", json={"message": ""})
    assert r.status_code == 400


def test_nlp_sentiment(client):
    r = client.post("/api/nlp/sentiment", json={"text": "good product great"})
    assert r.status_code == 200
    assert r.get_json()["success"]


def test_nlp_keywords(client):
    r = client.post("/api/nlp/keywords",
                    json={"text": "python programming data science"})
    assert r.get_json()["success"]


def test_nlp_word_count(client):
    r = client.post("/api/nlp/word-count", json={"text": "Hello World test"})
    body = r.get_json()
    assert body["success"]
    assert body["data"]["字符数"] == 16


def test_nlp_analyze(client):
    r = client.post("/api/nlp/analyze", json={"text": "Today is a great day"})
    assert r.get_json()["success"]


def test_config(client):
    body = client.get("/api/config").get_json()
    assert "deepseek_configured" in body


def test_chat_clear_and_history(client):
    assert client.post("/api/chat/clear", json={}).get_json()["success"]
    assert client.get("/api/chat/history").get_json()["success"]


def test_upload_csv_and_chart(client):
    data = {"file": (io.BytesIO(
        b"name,age,score,city\nAlice,25,88,Beijing\nBob,30,92,Shanghai"
    ), "test.csv")}
    r = client.post("/api/chat/upload", data=data,
                    content_type="multipart/form-data")
    body = r.get_json()
    assert body["success"]
    fp = body["data"]["file_path"]

    r = client.post("/api/chart/generate", json={"file_path": fp})
    body = r.get_json()
    assert body["success"]
    charts = body["data"]["charts"]
    assert len(charts) >= 3
    assert charts[0]["type"] == "histogram"
    assert len(charts[0]["image"]) > 1000
