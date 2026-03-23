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

    def test_health_model_field(self):
        resp = client.get("/health")
        assert "model_loaded" in resp.json()


class TestNEREndpoint:
    def test_ner(self):
        resp = client.post("/ner", json={"text": "John Smith works at Google in New York"})
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "tags" in data
        assert len(data["tags"]) > 0

    def test_ner_empty_string(self):
        """Empty text should fail validation (min_length=1)."""
        resp = client.post("/ner", json={"text": ""})
        assert resp.status_code == 422

    def test_ner_single_word(self):
        resp = client.post("/ner", json={"text": "Google"})
        assert resp.status_code == 200
        assert len(resp.json()["tags"]) == 1

    def test_ner_missing_text(self):
        """Missing required field should return 422."""
        resp = client.post("/ner", json={})
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_basic(self):
        resp = client.post("/ner/batch", json={"texts": ["John works at Google", "Visit Paris"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert "summary" in data

    def test_batch_single(self):
        resp = client.post("/ner/batch", json={"texts": ["Hello world"]})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1

    def test_batch_empty_list(self):
        """Empty list should fail validation."""
        resp = client.post("/ner/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_batch_contains_empty_string(self):
        """Batch containing empty strings should fail validation."""
        resp = client.post("/ner/batch", json={"texts": ["Hello", ""]})
        assert resp.status_code == 422


class TestTagsEndpoint:
    def test_list_tags(self):
        resp = client.get("/ner/tags")
        assert resp.status_code == 200
        tags = resp.json()
        assert len(tags) >= 5  # PER, ORG, LOC, MISC, DATE
        tag_names = [t["tag"] for t in tags]
        assert "PER" in tag_names
        assert "ORG" in tag_names

    def test_tags_have_descriptions(self):
        resp = client.get("/ner/tags")
        for tag_info in resp.json():
            assert "description" in tag_info
            assert len(tag_info["description"]) > 0
