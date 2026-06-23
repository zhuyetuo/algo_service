import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def client():
    with patch("main.init_db", new_callable=AsyncMock), \
         patch("main.close_db", new_callable=AsyncMock), \
         patch("main.start_scheduler"), \
         patch("main.stop_scheduler"):
        import main
        from fastapi.testclient import TestClient
        with TestClient(main.app) as c:
            yield c


def _make_session_mock():
    """Create a working AsyncSessionLocal context manager mock."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.return_value = mock_session
    return mock_cm


def _make_httpx_mock(status_code=200):
    """Create a working httpx.AsyncClient context manager mock."""
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=None)
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_http.post = AsyncMock(return_value=mock_resp)
    return mock_http


class TestHealthEndpoint:
    def test_health_all_ok(self, client):
        mock_cm = _make_session_mock()
        mock_http = _make_httpx_mock(200)

        with patch("main.AsyncSessionLocal", mock_cm), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = mock_http
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["mysql"] == "ok"
        assert body["tdengine"] == "ok"

    def test_health_postgres_down(self, client):
        # Mock AsyncSessionLocal to raise an exception
        mock_cm = MagicMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_cm.return_value = mock_session

        mock_http = _make_httpx_mock(200)

        with patch("main.AsyncSessionLocal", mock_cm), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = mock_http
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["mysql"] != "ok"

    def test_health_tdengine_down(self, client):
        mock_cm = _make_session_mock()

        # Mock httpx.AsyncClient to raise an exception on __aenter__
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("main.AsyncSessionLocal", mock_cm), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = mock_http
            response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["tdengine"] != "ok"
