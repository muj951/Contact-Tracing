import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.cluster import DBSCAN
from src.preprocessing import load_data

# ==========================================
# CHECKPOINT 2: TRAINING & LOGGING LOGIC
# ==========================================

def setup_mlflow():
    """Sets the tracking URI and ensures the default experiment exists."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "Contact_Tracing_Experiment"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def run_contact_tracing(df, input_name="Erin"):
    """
    Runs DBSCAN to find potential contacts for a given person.
    Logs parameters, metrics, and models to MLflow.
    """
    setup_mlflow()

    epsilon = 0.0018288  # Distance threshold
    min_samples = 2

    with mlflow.start_run(run_name=f"Run_{input_name}"):
        print(f"Running contact tracing training for: {input_name}...")

        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("user_of_interest", input_name)

        # Train DBSCAN
        model = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
        model.fit(df[['latitude', 'longitude']])
        df['cluster'] = model.labels_

        # Identify contacts
        input_name_clusters = df[df['id'] == input_name]['cluster'].unique().tolist()
        contacts = []
        for cluster in input_name_clusters:
            if cluster != -1:  # Ignore noise
                ids_in_cluster = df[df['cluster'] == cluster]['id'].unique()
                contacts.extend([m for m in ids_in_cluster if m != input_name])

        contacts = list(set(contacts))

        mlflow.log_metric("contacts_count", len(contacts))
        mlflow.sklearn.log_model(model, "dbscan_model")

        return contacts

# ==========================================
# CHECKPOINT 3: API INFERENCE LOGIC
# ==========================================

def get_latest_model():
    """
    Dynamically searches the local MLflow DB for the latest model run.
    """
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "Contact_Tracing_Experiment"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        raise Exception(f"Experiment '{experiment_name}' not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise Exception("No runs found in the experiment.")

    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/dbscan_model"
    print(f"Loading model from run: {latest_run_id}")

    return mlflow.sklearn.load_model(model_uri)

def predict_contacts(df, input_name, model):
    """
    Uses the loaded model to identify potential contacts.
    """
    df['cluster'] = model.labels_

    input_name_clusters = df[df['id'] == input_name]['cluster'].unique().tolist()
    contacts = []

    for cluster in input_name_clusters:
        if cluster != -1:
            ids_in_cluster = df[df['cluster'] == cluster]['id'].unique()
            contacts.extend([m for m in ids_in_cluster if m != input_name])

    return list(set(contacts))

# ==========================================
# MANUAL EXECUTION
# ==========================================

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        results = run_contact_tracing(df, "Erin")
        print(f"⚠️ Potential contacts identified: {results}")
