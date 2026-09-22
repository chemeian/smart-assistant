import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from app import app

with app.test_client() as c:
    r = c.get("/api/health")
    assert r.get_json()["status"] == "ok"
    print("[PASS] GET /api/health")

    r = c.get("/", follow_redirects=True)
    assert r.status_code == 200
    print("[PASS] GET /")

    r = c.get("/api/chat/providers")
    assert r.get_json()["success"]
    print("[PASS] GET /api/chat/providers")

    r = c.post("/api/chat/send", json={"message": "hi", "provider": "invalid"})
    assert r.status_code == 400
    print("[PASS] POST /api/chat/send invalid")

    r = c.post("/api/chat/send", json={"message": "", "provider": "deepseek"})
    assert r.status_code == 400
    print("[PASS] POST /api/chat/send empty")

    r = c.post("/api/nlp/sentiment", json={"text": "good product great"})
    d = r.get_json()
    assert d["success"]
    print(f"[PASS] sentiment -> {d['data']['sentiment']}")

    r = c.post("/api/nlp/sentiment", json={"text": "bad terrible awful"})
    d = r.get_json()
    print(f"[PASS] sentiment neg -> {d['data']['sentiment']}")

    r = c.post("/api/nlp/keywords", json={"text": "python programming data science"})
    d = r.get_json()
    assert d["success"]
    print(f"[PASS] keywords -> {d['data']['by_frequency'][0]['word']}")

    r = c.post("/api/nlp/summary", json={"text": "First of all this is the first sentence. This is the second sentence of the text. Third comes the third sentence with more words. Finally here is the fourth sentence of the text."})
    d = r.get_json()
    assert d["success"]
    print(f"[PASS] summary -> {d['data']['summary_sentences']} sentences")

    r = c.post("/api/nlp/word-count", json={"text": "Hello World test"})
    d = r.get_json()
    assert d["success"]
    assert d["data"]["characters"] == 16
    print(f"[PASS] word-count -> chars={d['data']['characters']} words={d['data']['words']}")

    r = c.post("/api/nlp/analyze", json={"text": "Today is a great day"})
    d = r.get_json()
    assert d["success"]
    print(f"[PASS] analyze -> sentiment={d['data']['sentiment']['sentiment']}")

    r = c.get("/api/config")
    assert "deepseek_configured" in r.get_json()
    print("[PASS] GET /api/config")

    r = c.post("/api/chat/clear", json={})
    assert r.get_json()["success"]
    print("[PASS] POST /api/chat/clear")

    r = c.get("/api/chat/history")
    assert r.get_json()["success"]
    print("[PASS] GET /api/chat/history")

    import io
    data = {"file": (io.BytesIO(
        b"name,age,score,city\nAlice,25,88,Beijing\nBob,30,92,Shanghai"
    ), "test.csv")}
    r = c.post("/api/chat/upload", data=data, content_type="multipart/form-data")
    d = r.get_json()
    assert d["success"]
    fp = d["data"]["file_path"]
    print(f"[PASS] POST /api/chat/upload -> file saved")

    r = c.post("/api/chart/generate", json={"file_path": fp})
    d = r.get_json()
    assert d["success"]
    assert len(d["data"]["charts"]) >= 3
    assert d["data"]["charts"][0]["type"] == "histogram"
    assert len(d["data"]["charts"][0]["image"]) > 1000
    print(f"[PASS] POST /api/chart/generate -> {len(d['data']['charts'])} charts, image {len(d['data']['charts'][0]['image'])} chars")

print("ALL TESTS PASSED")
