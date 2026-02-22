import pandas as pd
import mlflow
from src.train import run_contact_tracing

def test_run_contact_tracing_live_db():
    # Use a separate test database to keep your main data clean
    mlflow.set_tracking_uri("sqlite:///mlflow_test.db")

    data = {
        'id': ['UserA', 'UserA', 'UserB'],
        'latitude': [13.148, 13.149, 13.148],
        'longitude': [77.593, 77.593, 77.593],
        'timestamp': ['2020-07-04 15:35:30'] * 3
    }
    df = pd.DataFrame(data)

    contacts = run_contact_tracing(df, input_name="UserA")

    # Assertions for your logic
    assert "UserB" in contacts
    assert "UserA" not in contacts
