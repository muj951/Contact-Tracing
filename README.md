# 🛰️ Digital Contact Tracing via Machine Learning
**Repository:** [https://github.com/muj951/Contact-Tracing](https://github.com/muj951/Contact-Tracing)

## Group Members
- Mujahid Afzal
- Abdullah Zia
- Talha Siddique

## 1. Project Overview
Contact tracing is a process used by public health ministries to stop the spread of infectious diseases. This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for a digital tracing algorithm. It ingests location data, identifies potential physical contacts using machine learning, and serves these predictions through a robust, containerized REST API with an automated CI/CD lifecycle.

## 2. Problem Definition & Data
**Problem:** In the event of an outbreak, manually tracing contacts is slow and error-prone. We need an automated system to rapidly identify individuals who have been within a 6-foot radius of an infected patient during a 14-day window.
**Data:** The system utilizes `contact_tracking.json` containing user IDs, timestamps, and GPS coordinates.
**Modeling:** We apply **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to cluster geographical coordinates, identifying spatial-temporal proximity and potential exposure events.

## 3. System Architecture
The architecture is designed for reproducibility, modularity, and scalability:
- **Data & Processing Layer:** `src/preprocessing.py` ensures the JSON data is correctly ingested and parsed.
- **ML & Experiment Tracking:** `src/train.py` executes the clustering logic. MLflow tracks hyperparameters (Epsilon, min_samples) and saves model artifacts locally.
- **API Serving:** A FastAPI backend (`src/app.py`) loads the latest registered MLflow model to serve real-time predictions.
- **Containerization:** The application is packaged into an isolated Docker container.
- **CI/CD:** GitHub Actions automates testing and Docker image builds upon every push to the `main` branch.

## 4. MLOps Practices
This repository strictly enforces MLOps best practices:
- **Environment Management:** Managed via `uv` for high-performance dependency locking (`pyproject.toml` and `uv.lock`).
- **Version Control & Quality:** Pre-commit hooks run `ruff`, `trailing-whitespace`, and `end-of-file-fixer` on every commit.
- **Testing:** Unit tests are executed via `pytest`, utilizing `unittest.mock` to isolate API tests from the live database. We maintain **76% test coverage** (surpassing the 60% requirement).
- **Automation:** GitHub Actions serves as the Continuous Integration gatekeeper, preventing broken code from merging.

## 5. Monitoring & Reliability
**Monitoring Strategy:**
- **System Health:** Uvicorn and FastAPI provide native request logging (HTTP status codes, latency, error rates) for the `/trace` endpoint.
- **Model Tracking:** The MLflow UI (`uv run mlflow ui`) allows for comparative analysis of experiment runs and model versions.
- **Reliability:** Docker ensures environment consistency across development and production, while the CI pipeline prevents regressions by blocking failing pull requests.

## 6. Team Collaboration
Development was distributed across a 3-person team. Every member led a specific checkpoint phase while others supported via code reviews and collaboration. We enforced a feature-branch workflow where developers could not push directly to `main`. Every feature required a Pull Request, passing automated CI checks, and at least one manual peer review approval before merging.

## 7. Limitations & Future Work
- **Graph-Based Architecture:** The current DBSCAN model clusters direct spatial proximity (1st-degree contacts). Future iterations should map the contact data into a network using tools like GraphFrames. Applying Graph Theory algorithms like PageRank or connected components would allow the system to identify multi-degree transmission chains and super-spreaders.
- **Cloud Deployment:** Transitioning the Dockerized API from local hosting to a managed cloud service.
- **Streaming Data:** Replacing static data ingestion with a streaming architecture for real-time contact tracing.

---

## 🚀 How to Run Locally

Ensure you have [UV](https://github.com/astral-sh/uv) installed.

1. **Clone the project:**
   ```bash
   git clone [https://github.com/muj951/Contact-Tracing.git](https://github.com/muj951/Contact-Tracing.git)
   cd Contact-Tracing

🐳 Running the API via Docker
This project includes a fully containerized FastAPI application for model inference. The Docker build process automatically trains the model and saves the artifacts internally using MLflow.

1. Build the image: ```bash docker build -t contact-tracing-api . ```

2. Run the container: ```bash docker run -p 8000:8000 contact-tracing-api ```

3. Test the API: Open your browser and navigate to http://localhost:8000/docs to use the interactive Swagger UI. Test the /trace endpoint with a payload like {"user_name": "Judy"}.
