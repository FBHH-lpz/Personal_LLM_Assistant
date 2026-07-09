"""Tests for FastAPI routes."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    """Test admin endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Health check should return ok."""
        from httpx import ASGITransport, AsyncClient

        from app.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_root(self):
        """Root should return app info."""
        from httpx import ASGITransport, AsyncClient

        from app.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "Personal LLM Assistant"
            assert data["version"] == "2.0.0"


class TestConversationRoutes:
    """Test conversation CRUD."""

    @pytest.mark.asyncio
    async def test_list_conversations(self):
        """Listing conversations should return an empty list."""
        from httpx import ASGITransport, AsyncClient

        from app.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/conversations?user_id=test_user")
            assert resp.status_code == 200
            data = resp.json()
            assert "conversations" in data


class TestDocumentRoutes:
    """Test document upload validation."""

    @pytest.mark.asyncio
    async def test_upload_invalid_type(self):
        """Uploading an unsupported file type should return 400."""
        from httpx import ASGITransport, AsyncClient

        from app.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/documents/upload",
                files={"file": ("test.exe", b"binary content", "application/octet-stream")},
                data={"user_id": "test_user"},
            )
            assert resp.status_code == 400
            assert "Unsupported" in resp.json()["detail"]
