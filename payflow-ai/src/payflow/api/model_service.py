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
# PAYFLOW AI — MODEL SERVICE
#
# PART 5
#   Governance
#   Registry
#   @champion
#   failure threshold
#
# PART 6
#   FastAPI production inference
#
# PART 7
#   Docker MLflow
#   PostgreSQL
#   MinIO
#
# PART 8
#   Nested MLflow inference tracing
#
#
# Runtime trace:
#
# payflow.predict                  ← main.py
# │
# ├── mlflow.signature_validation ← this file
# │
# ├── sklearn.predict_proba       ← this file
# │
# └── failure.threshold           ← this file
#
# =========================================================


# =========================================================
# 1. CUSTOM SERVICE EXCEPTIONS
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
# 2. PAYFLOW MODEL SERVICE
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
        8. Emit nested MLflow inference spans
    """


    # =====================================================
    # INITIALIZATION
    # =====================================================

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


        mlflow.set_registry_uri(
            settings.tracking_uri
        )


        self.client = (
            MlflowClient(
                tracking_uri=(
                    settings.tracking_uri
                ),
                registry_uri=(
                    settings.tracking_uri
                ),
            )
        )


        # -------------------------------------------------
        # Runtime state.
        #
        # These values are populated by load().
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
    # 3. LOAD PRODUCTION MODEL
    # =====================================================

    def load(
        self,
    ) -> None:
        """
        Resolve the Registry alias and load the approved
        production model exactly once during API startup.

        This method does NOT train a model.

        It loads:

            models:/payflow-payment-success@champion
        """

        # Reset readiness while initialization is running.

        self.ready = False

        self.load_error = None


        try:

            # =============================================
            # 3.1 RESOLVE @champion
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
            # 3.2 READ MODEL VERSION TAGS
            #
            # These governance tags were created earlier
            # during the quality-gate / Registry workflow.
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
            # 3.3 GOVERNANCE CHECK
            #
            # @champion alone is NOT sufficient.
            #
            # The model must still prove:
            #
            #   quality_gate = PASSED
            #   registry_eligible = true
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
            # 3.4 READ VERSION-SPECIFIC FAILURE THRESHOLD
            #
            # IMPORTANT:
            #
            # Do NOT hard-code:
            #
            #     0.074343...
            #
            # Threshold belongs to the registered version.
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
            # 3.5 BUILD ALIAS-BASED MODEL URI
            #
            # Example:
            #
            # models:/payflow-payment-success@champion
            # =============================================

            model_uri = (
                self.settings
                .model_uri
            )


            # =============================================
            # 3.6 LOAD SKLEARN PIPELINE
            #
            # MLflow downloads model artifacts and rebuilds:
            #
            # preprocessing
            #     +
            # classifier
            # =============================================

            self.model = (
                mlflow_sklearn
                .load_model(
                    model_uri
                )
            )


            # =============================================
            # 3.7 LOAD MLFLOW MODEL METADATA
            #
            # ModelInfo contains the logged signature used
            # for runtime schema validation.
            # =============================================

            self.model_info = (
                mlflow.models
                .get_model_info(
                    model_uri
                )
            )


            # =============================================
            # 3.8 RECORD MODEL LOAD TIME
            # =============================================

            self.loaded_at_utc = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )


            # =============================================
            # 3.9 MODEL IS READY
            # =============================================

            self.ready = True


        except Exception as error:

            self.ready = False

            self.load_error = (
                str(
                    error
                )
            )

            raise


    # =====================================================
    # 4. REQUIRE READY
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
    # 5. BUILD + VALIDATE INFERENCE DATAFRAME
    # =====================================================

    def _build_frame(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        """
        Convert API records into the exact feature layout
        expected by the PayFlow sklearn Pipeline.

        This method also validates the dataframe against
        the MLflow model signature.
        """

        frame = (
            pd.DataFrame
            .from_records(
                records
            )
        )


        # -------------------------------------------------
        # 5.1 REQUIRED COLUMN VALIDATION
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
        # 5.2 REJECT UNSUPPORTED FIELDS
        #
        # Prevent accidental schema drift from entering the
        # model pipeline.
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
        # 5.3 ENFORCE TRAINING COLUMN ORDER
        # -------------------------------------------------

        frame = (
            frame[
                FEATURE_COLUMNS
            ]
            .copy()
        )


        # -------------------------------------------------
        # 5.4 MLFLOW SIGNATURE VALIDATION SPAN
        #
        # When this method is called inside:
        #
        #     payflow.predict
        #
        # this automatically becomes its child span.
        #
        # Trace:
        #
        # payflow.predict
        #   └── mlflow.signature_validation
        # -------------------------------------------------

        with mlflow.start_span(
            name="mlflow.signature_validation"
        ) as span:

            span.set_attributes(
                {
                    "payflow.feature_count":
                        len(
                            FEATURE_COLUMNS
                        ),

                    "payflow.record_count":
                        len(
                            frame
                        ),
                }
            )


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


                    span.set_outputs(
                        {
                            "schema_valid":
                                True,
                        }
                    )


                except MlflowException as error:

                    span.set_outputs(
                        {
                            "schema_valid":
                                False,
                        }
                    )


                    raise ModelContractError(
                        "Request does not match the "
                        "MLflow model signature. "
                        f"{error}"
                    ) from error


            else:

                # Model has no runtime input signature.
                #
                # We do not fail here because the original
                # service behavior allowed this condition.

                span.set_outputs(
                    {
                        "schema_valid":
                            True,

                        "signature_present":
                            False,
                    }
                )


        return frame


    # =====================================================
    # 6. PREDICT MANY RECORDS
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

        We use probabilities + the Registry threshold.

        We do NOT rely on:

            model.predict()

        because the production decision threshold may not
        be the classifier's default 0.50 threshold.
        """

        # -------------------------------------------------
        # Confirm startup completed successfully.
        # -------------------------------------------------

        self.require_ready()


        # -------------------------------------------------
        # Build exact inference dataframe.
        #
        # This also creates:
        #
        #     mlflow.signature_validation
        #
        # when a parent MLflow trace exists.
        # -------------------------------------------------

        frame = (
            self._build_frame(
                records
            )
        )


        assert (
            self.model
            is not None
        )


        assert (
            self.failure_threshold
            is not None
        )


        assert (
            self.model_version
            is not None
        )


        # =================================================
        # 6.1 MODEL PROBABILITY INFERENCE SPAN
        #
        # Parent trace:
        #
        # payflow.predict
        #
        # Child:
        #
        # sklearn.predict_proba
        #
        #
        # Existing helper functions are intentionally kept.
        #
        # They determine:
        #
        #     P(FAILED)  → class 0
        #     P(SUCCESS) → class 1
        #
        # =================================================

        with mlflow.start_span(
            name="sklearn.predict_proba"
        ) as span:

            span.set_attributes(
                {
                    "service.name":
                        "payflow-fastapi",

                    "mlflow.model.name":
                        self.settings
                        .registered_model_name,

                    "mlflow.model.version":
                        self.model_version,

                    "mlflow.model.alias":
                        self.settings
                        .model_alias,

                    "payflow.record_count":
                        len(
                            frame
                        ),

                    "payflow.feature_count":
                        len(
                            FEATURE_COLUMNS
                        ),
                }
            )


            # =============================================
            # P(FAILED)
            #
            # FAILED = class 0
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
            # P(SUCCESS)
            #
            # SUCCESS = class 1
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


            # -------------------------------------------------
            # Avoid logging every probability for large batches.
            #
            # Only operational summary information goes into the
            # span.
            # -------------------------------------------------

            span.set_outputs(
                {
                    "record_count":
                        len(
                            failure_probabilities
                        ),

                    "failure_probability_min":
                        float(
                            min(
                                failure_probabilities
                            )
                        ),

                    "failure_probability_max":
                        float(
                            max(
                                failure_probabilities
                            )
                        ),

                    "success_probability_min":
                        float(
                            min(
                                success_probabilities
                            )
                        ),

                    "success_probability_max":
                        float(
                            max(
                                success_probabilities
                            )
                        ),
                }
            )


        # =================================================
        # 6.2 VERSION-SPECIFIC THRESHOLD SPAN
        #
        # Parent:
        #
        # payflow.predict
        #
        # Child:
        #
        # failure.threshold
        #
        #
        # Rule:
        #
        # P(FAILED) >= threshold
        #       ↓
        # FAILED
        #
        # otherwise:
        #
        # SUCCESS
        # =================================================

        with mlflow.start_span(
            name="failure.threshold"
        ) as span:

            span.set_attributes(
                {
                    "service.name":
                        "payflow-fastapi",

                    "mlflow.model.name":
                        self.settings
                        .registered_model_name,

                    "mlflow.model.version":
                        self.model_version,

                    "mlflow.model.alias":
                        self.settings
                        .model_alias,

                    "payflow.failure_threshold":
                        float(
                            self.failure_threshold
                        ),

                    "payflow.threshold_probability":
                        "P(FAILED)",

                    "payflow.failed_class":
                        0,

                    "payflow.success_class":
                        1,

                    "payflow.record_count":
                        len(
                            frame
                        ),
                }
            )


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


            failed_count = sum(

                1

                for prediction
                in predictions

                if int(
                    prediction
                )
                == 0
            )


            success_count = (

                len(
                    predictions
                )

                - failed_count
            )


            span.set_outputs(
                {
                    "prediction_count":
                        len(
                            predictions
                        ),

                    "failed_count":
                        failed_count,

                    "success_count":
                        success_count,
                }
            )


        # =================================================
        # 6.3 BUILD API RESULT OBJECTS
        # =================================================

        results: list[
            dict[str, Any]
        ] = []


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
    # 7. OPERATIONAL MODEL METADATA
    # =====================================================

    def metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata for the currently loaded production
        model.

        Used by:

            GET /ready
            GET /v1/model
            prediction logs
            MLflow traces
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