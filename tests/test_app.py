from fastapi.testclient import TestClient
from unittest.mock import patch
from src.app import app

client = TestClient(app)

@patch("src.app.predict_contacts")
def test_trace_endpoint_valid_user(mock_predict):
    """Test that the API returns a 200 success when the model works."""
    # Tell the fake model to just return "Bob"
    mock_predict.return_value = ["Bob"]

    response = client.post("/trace", json={"user_name": "Judy"})
    assert response.status_code == 200

    data = response.json()
    assert data["user_of_interest"] == "Judy"
    assert data["potential_contacts"] == ["Bob"]

@patch("src.app.predict_contacts")
def test_trace_endpoint_invalid_user(mock_predict):
    """Test that the API handles errors gracefully."""
    # Force the fake model to throw an error like it did with the dummy data
    mock_predict.side_effect = Exception("Length of values mismatch")

    response = client.post("/trace", json={"user_name": "GhostUser99"})

    # Since your app catches Exceptions and throws a 400 or 404, we test for that
    assert response.status_code in [400, 404]
