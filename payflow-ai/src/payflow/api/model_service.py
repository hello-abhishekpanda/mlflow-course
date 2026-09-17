from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from typing import Any

import mlflow
import mlflow.models
import mlflow.sklearn as mlflow_sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from payflow.api.config import (
    Settings,
)

from payflow.features.payment_success import (
    FEATURE_COLUMNS,
)

from payflow.models.thresholding import (
    get_class_probability,
    predict_with_failure_threshold,
)


# =========================================================
# CUSTOM SERVICE EXCEPTIONS
# =========================================================

class ModelNotReadyError(
    RuntimeError
):
    """
    Raised when prediction is attempted before the
    production model has been loaded.
    """


class ModelContractError(
    ValueError
):
    """
    Raised when incoming inference data does not satisfy
    the MLflow model signature.
    """


class ModelGovernanceError(
    RuntimeError
):
    """
    Raised when the Registry says the model should not be
    used for production inference.
    """


# =========================================================
# PAYFLOW MODEL SERVICE
# =========================================================

class PayFlowModelService:
    """
    Owns the currently loaded production model.

    Responsibilities:

        1. Resolve @champion from MLflow Registry
        2. Verify governance metadata
        3. Read model-specific failure threshold
        4. Load sklearn Pipeline
        5. Validate inference schema
        6. Calculate probabilities
        7. Apply threshold policy
    """


    def __init__(
        self,
        settings: Settings,
    ) -> None:

        self.settings = (
            settings
        )


        # -------------------------------------------------
        # Configure MLflow before creating the Registry
        # client.
        # -------------------------------------------------

        mlflow.set_tracking_uri(
            settings.tracking_uri
        )


        self.client = (
            MlflowClient()
        )


        # -------------------------------------------------
        # Runtime state.
        # -------------------------------------------------

        self.model: Any | None = (
            None
        )


        self.model_info: Any | None = (
            None
        )


        self.model_version: str | None = (
            None
        )


        self.model_tags: dict[
            str,
            str,
        ] = {}


        self.failure_threshold: float | None = (
            None
        )


        self.loaded_at_utc: str | None = (
            None
        )


        self.ready = False


        self.load_error: str | None = (
            None
        )


    # =====================================================
    # LOAD PRODUCTION MODEL
    # =====================================================

    def load(
        self,
    ) -> None:
        """
        Resolve the Registry alias and load the approved
        production model exactly once during API startup.
        """

        # Reset readiness while loading.
        self.ready = False

        self.load_error = None


        try:

            # =============================================
            # 1. RESOLVE @champion
            # =============================================

            model_version = (
                self.client
                .get_model_version_by_alias(

                    name=(
                        self.settings
                        .registered_model_name
                    ),

                    alias=(
                        self.settings
                        .model_alias
                    ),
                )
            )


            self.model_version = str(
                model_version.version
            )


            # =============================================
            # 2. READ VERSION TAGS
            #
            # These were created during Part 5.
            # =============================================

            self.model_tags = dict(
                model_version.tags
                or {}
            )


            quality_gate = (
                self.model_tags.get(
                    "quality_gate"
                )
            )


            registry_eligible = (
                self.model_tags.get(
                    "registry_eligible"
                )
            )


            # =============================================
            # 3. GOVERNANCE CHECK
            #
            # Even though the alias says @champion, we also
            # require the version metadata to prove that the
            # model passed its production gate.
            # =============================================

            if quality_gate != "PASSED":

                raise ModelGovernanceError(
                    "Production model does not have "
                    "quality_gate=PASSED."
                )


            if (
                str(
                    registry_eligible
                ).lower()
                != "true"
            ):

                raise ModelGovernanceError(
                    "Production model does not have "
                    "registry_eligible=true."
                )


            # =============================================
            # 4. READ FAILURE THRESHOLD
            #
            # Threshold belongs to THIS registered version.
            #
            # We deliberately do not hard-code 0.074343.
            # =============================================

            threshold_text = (
                self.model_tags.get(
                    "failure_threshold"
                )
            )


            if threshold_text is None:

                raise ModelGovernanceError(
                    "Registered model version has no "
                    "failure_threshold tag."
                )


            threshold = float(
                threshold_text
            )


            if not 0.0 < threshold < 1.0:

                raise ModelGovernanceError(
                    "failure_threshold must be between "
                    "0 and 1."
                )


            self.failure_threshold = (
                threshold
            )


            # =============================================
            # 5. BUILD ALIAS URI
            # =============================================

            model_uri = (
                self.settings
                .model_uri
            )


            # =============================================
            # 6. LOAD SKLEARN PIPELINE
            #
            # This downloads the registered model artifacts
            # and reconstructs:
            #
            #     preprocessing
            #         +
            #     classifier
            # =============================================

            self.model = (
                mlflow_sklearn
                .load_model(
                    model_uri
                )
            )


            # =============================================
            # 7. LOAD MLFLOW MODEL METADATA
            #
            # We use ModelInfo for the model signature.
            # =============================================

            self.model_info = (
                mlflow.models
                .get_model_info(
                    model_uri
                )
            )


            # =============================================
            # 8. RECORD LOAD TIME
            # =============================================

            self.loaded_at_utc = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )


            # =============================================
            # 9. MODEL IS READY
            # =============================================

            self.ready = True


        except Exception as error:

            self.ready = False

            self.load_error = (
                str(error)
            )

            raise


    # =====================================================
    # REQUIRE READY
    # =====================================================

    def require_ready(
        self,
    ) -> None:
        """
        Prevent predictions while model initialization has
        not completed successfully.
        """

        if (
            not self.ready
            or self.model is None
            or self.failure_threshold is None
        ):

            raise ModelNotReadyError(
                self.load_error
                or
                "Model is not ready."
            )


    # =====================================================
    # BUILD DATAFRAME
    # =====================================================

    def _build_frame(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        """
        Convert API records into the exact feature layout
        expected by the PayFlow pipeline.
        """

        frame = (
            pd.DataFrame
            .from_records(
                records
            )
        )


        # -------------------------------------------------
        # Verify required columns.
        # -------------------------------------------------

        missing_columns = (

            set(
                FEATURE_COLUMNS
            )

            - set(
                frame.columns
            )
        )


        if missing_columns:

            raise ModelContractError(
                "Missing inference features: "
                f"{sorted(missing_columns)}"
            )


        # -------------------------------------------------
        # Reject accidental unsupported fields before they
        # reach the ML pipeline.
        # -------------------------------------------------

        extra_columns = (

            set(
                frame.columns
            )

            - set(
                FEATURE_COLUMNS
            )
        )


        if extra_columns:

            raise ModelContractError(
                "Unsupported inference fields: "
                f"{sorted(extra_columns)}"
            )


        # -------------------------------------------------
        # Enforce training column ordering.
        # -------------------------------------------------

        frame = (
            frame[
                FEATURE_COLUMNS
            ]
            .copy()
        )


        # -------------------------------------------------
        # Validate against the MLflow model signature.
        #
        # The signature was logged during model training.
        # -------------------------------------------------

        if (
            self.model_info
            is not None
            and
            self.model_info.signature
            is not None
            and
            self.model_info.signature.inputs
            is not None
        ):

            try:

                mlflow.models.validate_schema(

                    frame,

                    self.model_info
                    .signature
                    .inputs,
                )

            except MlflowException as error:

                raise ModelContractError(
                    "Request does not match the "
                    "MLflow model signature. "
                    f"{error}"
                ) from error


        return frame


    # =====================================================
    # PREDICT MANY RECORDS
    # =====================================================

    def predict_records(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:
        """
        Score one or more PayFlow transactions.

        IMPORTANT:

        We use predict_proba(), not model.predict().

        Why?

        The business decision threshold is stored in the
        Model Registry and may be different from 0.50.
        """

        self.require_ready()


        frame = (
            self._build_frame(
                records
            )
        )


        assert self.model is not None

        assert (
            self.failure_threshold
            is not None
        )


        # =============================================
        # 1. P(FAILED)
        # =============================================

        failure_probabilities = (
            get_class_probability(

                model=(
                    self.model
                ),

                X=(
                    frame
                ),

                class_label=0,
            )
        )


        # =============================================
        # 2. P(SUCCESS)
        # =============================================

        success_probabilities = (
            get_class_probability(

                model=(
                    self.model
                ),

                X=(
                    frame
                ),

                class_label=1,
            )
        )


        # =============================================
        # 3. APPLY VERSION-SPECIFIC THRESHOLD
        # =============================================

        predictions = (
            predict_with_failure_threshold(

                failure_probability=(
                    failure_probabilities
                ),

                threshold=(
                    self.failure_threshold
                ),
            )
        )


        results = []


        for (
            prediction,
            failure_probability,
            success_probability,
        ) in zip(

            predictions,

            failure_probabilities,

            success_probabilities,

            strict=True,
        ):

            prediction_code = int(
                prediction
            )


            prediction_label = (

                "FAILED"

                if prediction_code == 0

                else "SUCCESS"
            )


            results.append(
                {
                    "prediction":
                        prediction_label,

                    "prediction_code":
                        prediction_code,

                    "failure_probability":
                        float(
                            failure_probability
                        ),

                    "success_probability":
                        float(
                            success_probability
                        ),
                }
            )


        return results


    # =====================================================
    # OPERATIONAL MODEL METADATA
    # =====================================================

    def metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata for the currently loaded production
        model.
        """

        self.require_ready()


        assert (
            self.model_version
            is not None
        )


        assert (
            self.failure_threshold
            is not None
        )


        assert (
            self.loaded_at_utc
            is not None
        )


        return {

            "model_name":
                self.settings
                .registered_model_name,

            "model_alias":
                self.settings
                .model_alias,

            "model_version":
                self.model_version,

            "model_uri":
                self.settings
                .model_uri,

            "source_model_id":
                self.model_tags.get(
                    "source_logged_model_id"
                ),

            "quality_gate":
                self.model_tags.get(
                    "quality_gate",
                    "UNKNOWN",
                ),

            "registry_eligible":
                (
                    str(
                        self.model_tags.get(
                            "registry_eligible"
                        )
                    ).lower()
                    == "true"
                ),

            "failure_threshold":
                self.failure_threshold,

            "loaded_at_utc":
                self.loaded_at_utc,
        }