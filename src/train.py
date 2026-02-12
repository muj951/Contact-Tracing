import mlflow
from sklearn.cluster import DBSCAN
from src.preprocessing import load_data

def setup_mlflow():
    """Sets the tracking URI and ensures the default experiment exists."""
    # Use a local SQLite database for tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Check if experiment exists, if not, create it to avoid ID 0 errors
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

    # --- Checkpoint 2: Define and log hyperparameters ---
    epsilon = 0.0018288  # Distance threshold
    min_samples = 2

    with mlflow.start_run(run_name=f"Run_{input_name}"):
        print(f"Running contact tracing for: {input_name}...")

        # Log Parameters to MLflow
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("user_of_interest", input_name)

        # Cluster the coordinates using Haversine distance
        model = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
        model.fit(df[['latitude', 'longitude']])
        df['cluster'] = model.labels_

        # Identify clusters associated with the input_name
        input_name_clusters = df[df['id'] == input_name]['cluster'].unique().tolist()
        infected_names = []

        for cluster in input_name_clusters:
            if cluster != -1:  # Ignore noise points
                ids_in_cluster = df[df['cluster'] == cluster]['id'].unique()
                for member_id in ids_in_cluster:
                    if member_id != input_name:
                        infected_names.append(member_id)

        contacts = list(set(infected_names))

        # Log Metrics to MLflow
        mlflow.log_metric("contacts_count", len(contacts))

        # Log the Model Artifact
        mlflow.sklearn.log_model(model, "dbscan_model")

        return contacts

if __name__ == "__main__":
    # Load data using the modular preprocessing function
    df = load_data()
    if df is not None:
        results = run_contact_tracing(df, "Erin")
        print(f"⚠️ Potential contacts identified: {results}")
