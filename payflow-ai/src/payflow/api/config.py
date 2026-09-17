from __future__ import annotations

import os

from dataclasses import dataclass

from dotenv import load_dotenv


# =========================================================
# LOAD .env
# =========================================================

load_dotenv()


# =========================================================
# API SETTINGS
#
# Keeping environment access here prevents model-serving
# code from being scattered with os.getenv() calls.
# =========================================================

@dataclass(
    frozen=True,
    slots=True,
)
class Settings:
    """
    Immutable configuration used by the PayFlow API.
    """

    tracking_uri: str

    registered_model_name: str

    model_alias: str

    api_title: str

    api_version: str


    @property
    def model_uri(
        self,
    ) -> str:
        """
        Build the registry alias URI.

        Example:

            models:/payflow-payment-success@champion
        """

        return (
            f"models:/"
            f"{self.registered_model_name}"
            f"@{self.model_alias}"
        )


# =========================================================
# CREATE SETTINGS
# =========================================================

def load_settings() -> Settings:
    """
    Read runtime settings from environment variables.
    """

    return Settings(

        tracking_uri=(
            os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            )
        ),

        registered_model_name=(
            os.getenv(
                "PAYFLOW_REGISTERED_MODEL_NAME",
                "payflow-payment-success",
            )
        ),

        model_alias=(
            os.getenv(
                "PAYFLOW_MODEL_ALIAS",
                "champion",
            )
        ),

        api_title=(
            "PayFlow AI Prediction API"
        ),

        api_version=(
            "1.0.0"
        ),
    )