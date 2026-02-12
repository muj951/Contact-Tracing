import pandas as pd
from src.preprocessing import load_data

def test_load_data_returns_dataframe():
    """Test that load_data returns a valid pandas DataFrame."""
    df = load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_data_columns():
    """Test that the loaded data contains the required columns."""
    df = load_data()
    expected_columns = ['id', 'timestamp', 'latitude', 'longitude']
    for col in expected_columns:
        assert col in df.columns
