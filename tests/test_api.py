"""
Tests — CI chạy file này trước khi build Docker images.
Nếu tests fail → không build → không deploy code lỗi.
"""
import pytest


def test_api_import():
    """Test: serving code import được không."""
    from api_gateway.main import app
    assert app is not None


def test_health_endpoint():
    """Test: health endpoint trả ok."""
    from fastapi.testclient import TestClient
    from api_gateway.main import app
    
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_infer_cold_user():
    """Test: cold user nhận popular items."""
    from fastapi.testclient import TestClient
    from api_gateway.main import app, load_artifacts
    import api_gateway.main as api_module
    
    # Mock popular items
    api_module.POPULAR_ITEMS = ["item_1", "item_2", "item_3"]
    api_module.USER2ID = {"warm_user": 0}
    api_module.MODEL = "mock"  # just needs to be not None
    
    client = TestClient(app)
    response = client.get("/infer?user_id=unknown_user&k=2")
    assert response.status_code == 200
    data = response.json()
    assert data["is_warm_user"] == False
    assert len(data["recommendations"]) == 2


def test_infer_warm_user():
    """Test: warm user detected correctly."""
    from fastapi.testclient import TestClient
    from api_gateway.main import app
    import api_gateway.main as api_module
    
    api_module.POPULAR_ITEMS = ["item_1", "item_2"]
    api_module.USER2ID = {"warm_user": 0}
    api_module.MODEL = "mock"
    
    client = TestClient(app)
    response = client.get("/infer?user_id=warm_user&k=2")
    assert response.status_code == 200
    assert response.json()["is_warm_user"] == True