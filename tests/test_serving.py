from fastapi.testclient import TestClient

from src.serving.api import create_app


def test_health_endpoints():
    with TestClient(create_app()) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health/startup").status_code == 200


def test_recommend_requires_auth():
    with TestClient(create_app()) as client:
        r = client.get("/recommend/0")
        assert r.status_code == 401


def test_recommend_valid(prepare_artifacts):
    with TestClient(create_app()) as client:
        r = client.get(
            "/recommend/0",
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == 0
        assert isinstance(body["recommendations"], list)
        assert "request_id" in body


def test_recommend_invalid_user(prepare_artifacts):
    with TestClient(create_app()) as client:
        r = client.get(
            "/recommend/999999",
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 400


def test_batch_recommend(prepare_artifacts):
    with TestClient(create_app()) as client:
        r = client.post(
            "/recommend/batch",
            json={"user_ids": [0, 1, 2]},
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert set(body["results"].keys()) == {"0", "1", "2"}


def test_security_headers_present():
    with TestClient(create_app()) as client:
        r = client.get("/health/live")
        for header in ["X-Content-Type-Options", "X-Frame-Options",
                       "Referrer-Policy"]:
            assert header in r.headers


def test_request_id_echoed():
    with TestClient(create_app()) as client:
        r = client.get("/health/live", headers={"X-Request-ID": "abc123"})
        assert r.headers.get("X-Request-ID") == "abc123"
