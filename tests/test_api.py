import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "tasks" in data


def test_list_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "hate" in data
    assert "support" in data


def test_invalid_task_rejection():
    response = client.post(
        "/analyze",
        json={
            "text": "Hello world",
            "tasks": ["non_existent_task"],
        },
    )
    assert response.status_code == 400
    assert "Unknown task" in response.json()["detail"]


def test_custom_task_validation():
    # Options must have at least 2 items
    response = client.post(
        "/analyze",
        json={
            "text": "Hello world",
            "custom_tasks": {
                "invalid": {
                    "question": "Choose one?",
                    "options": ["only_one"],
                }
            },
        },
    )
    assert response.status_code == 422
