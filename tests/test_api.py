"""Tests for API."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestNEREndpoint:
    def test_ner(self):
        resp = client.post("/ner", json={"text": "John Smith works at Google in New York"})
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "tags" in data
        assert len(data["tags"]) > 0
