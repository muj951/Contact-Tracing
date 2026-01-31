import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path
from preprocessing import load_data

def run_contact_tracing(df, input_name="Erin"):
    """
    Runs DBSCAN clustering to find potential contacts.
    """
    print(f"🔄 Running contact tracing for: {input_name}...")
    
    # epsilon = radial distance of 6 feet in kilometers
    epsilon = 0.0048288 
    
    # Initialize and fit DBSCAN
    model = DBSCAN(eps=epsilon, min_samples=2, metric='haversine')
    
    # We use latitude and longitude for clustering
    model.fit(df[['latitude', 'longitude']])
    df['cluster'] = model.labels_
    
    # Find the cluster IDs associated with the input_name
    input_name_clusters = df[df['id'] == input_name]['cluster'].unique().tolist()
    
    infected_names = []
    
    # If the person is in a cluster (and not noise -1), find others in that same cluster
    for cluster in input_name_clusters:
        if cluster != -1:
            ids_in_cluster = df[df['cluster'] == cluster]['id'].unique()
            for member_id in ids_in_cluster:
                if member_id != input_name:
                    infected_names.append(member_id)
    
    return list(set(infected_names))

if __name__ == "__main__":
    # 1. Load data using our path-friendly function
    df = load_data()
    
    if df is not None:
        # 2. Run the model
        potential_contacts = run_contact_tracing(df, "Erin")
        
        # 3. Output results
        if potential_contacts:
            print(f" Potential contacts found: {potential_contacts}")
        else:
            print(" No close contacts identified.")