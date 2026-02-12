import pandas as pd
from src.train import run_contact_tracing

def test_run_contact_tracing_finds_contacts():
    # Create a small dummy dataset
    data = {
        'id': ['UserA', 'UserA', 'UserB'],
        'latitude': [13.148, 13.149, 13.148], # UserA and UserB are very close
        'longitude': [77.593, 77.593, 77.593],
        'timestamp': ['2020-07-04 15:35:30'] * 3
    }
    df = pd.DataFrame(data)

    # Run the function
    contacts = run_contact_tracing(df, input_name="UserA")

    # Assertions
    assert "UserB" in contacts
    assert "UserA" not in contacts  # Should not find themselves as a contact

def test_run_contact_tracing_no_contacts():
    # Create data where users are far apart
    data = {
        'id': ['UserA', 'UserC'],
        'latitude': [13.148, 14.000],
        'longitude': [77.593, 78.000],
        'timestamp': ['2020-07-04 15:35:30'] * 2
    }
    df = pd.DataFrame(data)

    contacts = run_contact_tracing(df, input_name="UserA")
    assert len(contacts) == 0
