import pandas as pd
from pathlib import Path

# Universal path to the data file 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "contact_tracking.json"

def load_data(filepath=DATA_PATH):
    """Loads JSON data using a path-agnostic approach."""
    try:
        if not filepath.exists():
            print(f" Error: File not found at {filepath}")
            return None
            
        df = pd.read_json(filepath)
        print(f" Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f" Unexpected error loading data: {e}")
        return None

if __name__ == "__main__":
    print("Running preprocessing check...")
    df = load_data()
    if df is not None:
        print(df.head()) # This shows the first 5 rows