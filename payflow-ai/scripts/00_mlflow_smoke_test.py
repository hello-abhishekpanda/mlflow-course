from pathlib import Path

import mlflow


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "payflow-setup"


mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

artifact_dir = Path("artifacts")
artifact_dir.mkdir(exist_ok=True)

artifact_file = artifact_dir / "setup.txt"

artifact_file.write_text(
    "PayFlow AI MLflow setup is working.",
    encoding="utf-8",
)

with mlflow.start_run(run_name="mlflow-smoke-test") as run:

    mlflow.log_params(
        {
            "project": "payflow-ai",
            "domain": "upi-payments",
            "currency": "INR",
            "environment": "local",
        }
    )

    mlflow.log_metric(
        "setup_ok",
        1.0,
    )

    mlflow.set_tags(
        {
            "course": "payflow-ai-mlflow",
            "stage": "setup",
        }
    )

    mlflow.log_artifact(
        str(artifact_file),
        artifact_path="setup",
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment: {EXPERIMENT_NAME}")