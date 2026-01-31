# 🛰️ Digital Contact Tracing via Machine Learning
**Repository:** [https://github.com/muj951/Contact-Tracing](https://github.com/muj951/Contact-Tracing)

## Group Members
- Mujahid Afzal
- Abdullah Zia
- Talha Siddique

## Every Member will do one checkpoint and others will support in commit, pull request and collaburate during the entire project.

## 1. Project Overview
Contact tracing is a process used by public health ministries to stop the spread of infectious diseases like COVID-19. This project implements a digital tracing algorithm using GPS data and the **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) algorithm.

By clustering geographical coordinates, we can identify individuals who have been within a 6-foot radius of an infected patient during the last 14 days.

## 2. Project Setup (Checkpoint 1)
This repository is configured following MLOps best practices:
- **Environment Management:** Managed via `uv` for high-performance dependency locking.
- **Reproducibility:** All paths are handled via `pathlib` for cross-platform compatibility.
- **Modularity:** Separate modules for data loading (`preprocessing.py`) and model execution (`train.py`).


## 3. System Architecture
- **Data Layer:** `contact_tracking.json` containing user IDs, timestamps, and GPS coordinates.
- **Processing Layer:** `src/preprocessing.py` ensures the data is correctly ingested.
- **ML Layer:** `src/train.py` executes the DBSCAN clustering logic.
- **Dependency Layer:** `pyproject.toml` and `uv.lock` ensure the environment is identical on every machine.

## 4. How to Run
Ensure you have [UV](https://github.com/astral-sh/uv) installed.

1. **Clone the project:**
   ```bash
   git clone [https://github.com/muj951/Contact-Tracing.git](https://github.com/muj951/Contact-Tracing.git)
   cd Contact-Tracing
