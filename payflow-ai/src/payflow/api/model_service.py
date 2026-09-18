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
#   Version-specific failure threshold
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
#   MLflow tracing
#   Nested inference spans
#   Runtime model-contract enforcement
#
#
# Trace hierarchy:
#
# payflow.predict                     ← main.py
# │
# ├── mlflow.signature_validation     ← this file
# │
# ├── sklearn.predict_proba           ← this file
# │
# └── failure.threshold               ← this file
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
    production model has been loaded successfully.
    """


class ModelContractError(
    ValueError
):
    """
    Raised when inference input does not satisfy the
    production model contract.
    """


class ModelGovernanceError(
    RuntimeError
):
    """
    Raised when Registry metadata says that a model must
    not be used for production inference.
    """


# =========================================================
# 2. PAYFLOW MODEL SERVICE
# =========================================================


class PayFlowModelService:
    """
    Owns the currently loaded production PayFlow model.

    Responsibilities:

        1. Resolve @champion from MLflow Registry
        2. Verify model governance tags
        3. Read version-specific failure threshold
        4. Load sklearn Pipeline
        5. Load MLflow model signature
        6. Build exact inference dataframe
        7. Normalize runtime dtypes
        8. Validate MLflow signature
        9. Calculate P(FAILED) / P(SUCCESS)
       10. Apply failure threshold
       11. Return production prediction
       12. Emit nested MLflow spans
    """


    # =====================================================
    # 3. INITIALIZATION
    # =====================================================

    def __init__(
        self,
        settings: Settings,
    ) -> None:

        self.settings = (
            settings
        )


        # -------------------------------------------------
        # MLflow must be configured before Registry access.
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
        # Runtime model state
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
    # 4. LOAD PRODUCTION MODEL
    # =====================================================

    def load(
        self,
    ) -> None:
        """
        Resolve and load the approved production model.

        This method does NOT train a model.

        Production URI:

            models:/payflow-payment-success@champion
        """

        self.ready = False

        self.load_error = None


        try:

            # =============================================
            # 4.1 RESOLVE @champion
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
            # 4.2 MODEL VERSION TAGS
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
            # 4.3 GOVERNANCE ENFORCEMENT
            #
            # @champion alone is not sufficient.
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
            # 4.4 FAILURE THRESHOLD
            #
            # Never hard-code the production threshold.
            #
            # It belongs to the registered model version.
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


            try:

                threshold = float(
                    threshold_text
                )

            except (
                TypeError,
                ValueError,
            ) as error:

                raise ModelGovernanceError(
                    "failure_threshold is not numeric."
                ) from error


            if not (
                0.0
                < threshold
                < 1.0
            ):

                raise ModelGovernanceError(
                    "failure_threshold must be between "
                    "0 and 1."
                )


            self.failure_threshold = (
                threshold
            )


            # =============================================
            # 4.5 MODEL URI
            # =============================================

            model_uri = (
                self.settings
                .model_uri
            )


            # =============================================
            # 4.6 LOAD SKLEARN PIPELINE
            #
            # Artifacts:
            #
            # MLflow
            #   ↓
            # MinIO
            #   ↓
            # sklearn Pipeline
            # =============================================

            self.model = (
                mlflow_sklearn
                .load_model(
                    model_uri
                )
            )


            # =============================================
            # 4.7 LOAD MLFLOW MODELINFO
            #
            # Contains the production model signature.
            # =============================================

            self.model_info = (
                mlflow.models
                .get_model_info(
                    model_uri
                )
            )


            # =============================================
            # 4.8 RECORD LOAD TIME
            # =============================================

            self.loaded_at_utc = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )


            # =============================================
            # 4.9 SERVICE IS READY
            # =============================================

            self.ready = True


        except Exception as error:

            self.ready = False

            self.load_error = str(
                error
            )

            raise


    # =====================================================
    # 5. REQUIRE READY
    # =====================================================

    def require_ready(
        self,
    ) -> None:
        """
        Prevent prediction traffic until startup model
        initialization has completed successfully.
        """

        if (
            not self.ready
            or self.model is None
            or self.failure_threshold is None
            or self.model_version is None
        ):

            raise ModelNotReadyError(
                self.load_error
                or
                "Model is not ready."
            )


    # =====================================================
    # 6. NORMALIZE INTEGER CONTRACT COLUMN
    # =====================================================

    @staticmethod
    def _normalize_int64_column(
        frame: pd.DataFrame,
        column_name: str,
    ) -> None:
        """
        Normalize one inference column to int64 safely.

        Accepted examples:

            1250
            1250.0
            "1250"

        Rejected examples:

            1250.75
            "abc"
            null

        This matters because the current registered model
        signature expects MLflow `long`, which maps to
        pandas / NumPy int64.

        We intentionally DO NOT truncate decimal amounts.
        """

        try:

            numeric = pd.to_numeric(
                frame[
                    column_name
                ],
                errors="raise",
            )


        except (
            TypeError,
            ValueError,
        ) as error:

            raise ModelContractError(
                f"{column_name} must be an integer-compatible "
                "numeric value."
            ) from error


        # -------------------------------------------------
        # Missing values are invalid for required features.
        # -------------------------------------------------

        if numeric.isna().any():

            raise ModelContractError(
                f"{column_name} contains a null value."
            )


        # -------------------------------------------------
        # The production signature expects int64.
        #
        # A value such as 1250.75 must NOT silently become
        # 1250.
        # -------------------------------------------------

        non_integral_mask = (

            numeric.astype(
                "float64"
            )

            % 1

            != 0
        )


        if non_integral_mask.any():

            invalid_values = (
                numeric[
                    non_integral_mask
                ]
                .tolist()
            )


            raise ModelContractError(
                f"{column_name} must contain whole-number "
                f"values because the production MLflow "
                f"signature expects long/int64. "
                f"Invalid values: {invalid_values}"
            )


        try:

            frame[
                column_name
            ] = (
                numeric.astype(
                    "int64"
                )
            )


        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:

            raise ModelContractError(
                f"{column_name} could not be converted "
                "safely to int64."
            ) from error


    # =====================================================
    # 7. BUILD + VALIDATE INFERENCE DATAFRAME
    # =====================================================

    def _build_frame(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        """
        Convert incoming API records into the exact runtime
        dataframe expected by the production pipeline.

        Steps:

            records
              ↓
            DataFrame
              ↓
            required-column validation
              ↓
            extra-column validation
              ↓
            FEATURE_COLUMNS ordering
              ↓
            dtype normalization
              ↓
            MLflow signature validation
        """

        # -------------------------------------------------
        # Empty inference requests should never reach the
        # model.
        # -------------------------------------------------

        if not records:

            raise ModelContractError(
                "At least one inference record is required."
            )


        # =============================================
        # 7.1 BUILD DATAFRAME
        # =============================================

        frame = (
            pd.DataFrame
            .from_records(
                records
            )
        )


        # =============================================
        # 7.2 REQUIRED FEATURES
        # =============================================

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


        # =============================================
        # 7.3 REJECT EXTRA FEATURES
        # =============================================

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


        # =============================================
        # 7.4 ENFORCE TRAINING COLUMN ORDER
        # =============================================

        frame = (
            frame[
                FEATURE_COLUMNS
            ]
            .copy()
        )


        # =============================================
        # 7.5 NORMALIZE NUMERIC DTYPES
        #
        # Current MLflow signature expects:
        #
        # amount_inr
        #     long / int64
        #
        # hour_of_day
        #     long / int64
        #
        # is_weekend
        #     long / int64
        #
        # JSON/Pydantic/Pandas can otherwise create a
        # float64 or object dtype.
        # =============================================

        self._normalize_int64_column(
            frame,
            "amount_inr",
        )

        self._normalize_int64_column(
            frame,
            "hour_of_day",
        )

        self._normalize_int64_column(
            frame,
            "is_weekend",
        )


        # =============================================
        # 7.6 NORMALIZE CATEGORICAL FEATURES
        #
        # The model signature expects strings for these
        # columns.
        # =============================================

        categorical_columns = [

            column

            for column
            in FEATURE_COLUMNS

            if column not in {
                "amount_inr",
                "hour_of_day",
                "is_weekend",
            }
        ]


        for column_name in categorical_columns:

            if frame[
                column_name
            ].isna().any():

                raise ModelContractError(
                    f"{column_name} contains a null value."
                )


            frame[
                column_name
            ] = (
                frame[
                    column_name
                ]
                .astype(
                    "string"
                )
            )


        # =============================================
        # 7.7 MLFLOW SIGNATURE VALIDATION SPAN
        #
        # When called from:
        #
        #     payflow.predict
        #
        # MLflow automatically nests this span:
        #
        # payflow.predict
        #   └── mlflow.signature_validation
        # =============================================

        with mlflow.start_span(
            name=(
                "mlflow.signature_validation"
            )
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

                    "payflow.amount_dtype":
                        str(
                            frame[
                                "amount_inr"
                            ].dtype
                        ),

                    "payflow.hour_dtype":
                        str(
                            frame[
                                "hour_of_day"
                            ].dtype
                        ),

                    "payflow.weekend_dtype":
                        str(
                            frame[
                                "is_weekend"
                            ].dtype
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

                            "signature_present":
                                True,
                        }
                    )


                except MlflowException as error:

                    span.set_outputs(
                        {
                            "schema_valid":
                                False,

                            "signature_present":
                                True,
                        }
                    )


                    raise ModelContractError(
                        "Request does not match the "
                        "MLflow model signature. "
                        f"{error}"
                    ) from error


            else:

                # -----------------------------------------
                # Preserve previous API behavior if an old
                # model exists without a logged signature.
                # -----------------------------------------

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
    # 8. PREDICT RECORDS
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

        Important:

        We use:

            predict_proba()

        plus the Registry threshold.

        We do NOT depend on:

            model.predict()

        because the governed production threshold is not
        necessarily 0.50.
        """

        self.require_ready()


        # =============================================
        # 8.1 BUILD + VALIDATE DATAFRAME
        #
        # Creates child span:
        #
        # mlflow.signature_validation
        # =============================================

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


        # =============================================
        # 8.2 PROBABILITY INFERENCE
        #
        # Trace:
        #
        # payflow.predict
        #   └── sklearn.predict_proba
        # =============================================

        with mlflow.start_span(
            name=(
                "sklearn.predict_proba"
            )
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


            # -----------------------------------------
            # FAILED is class 0.
            # -----------------------------------------

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


            # -----------------------------------------
            # SUCCESS is class 1.
            # -----------------------------------------

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


            if (
                len(
                    failure_probabilities
                )
                != len(
                    frame
                )
            ):

                raise RuntimeError(
                    "Failure probability output length "
                    "does not match input record count."
                )


            if (
                len(
                    success_probabilities
                )
                != len(
                    frame
                )
            ):

                raise RuntimeError(
                    "Success probability output length "
                    "does not match input record count."
                )


            # -----------------------------------------
            # Do not log all per-payment probabilities
            # for large batches.
            #
            # Log only operational summaries.
            # -----------------------------------------

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


        # =============================================
        # 8.3 APPLY GOVERNED FAILURE THRESHOLD
        #
        # Rule:
        #
        # P(FAILED) >= threshold
        #       ↓
        # FAILED
        #
        # otherwise
        #
        # SUCCESS
        #
        #
        # Trace:
        #
        # payflow.predict
        #   └── failure.threshold
        # =============================================

        with mlflow.start_span(
            name=(
                "failure.threshold"
            )
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


            if (
                len(
                    predictions
                )
                != len(
                    frame
                )
            ):

                raise RuntimeError(
                    "Prediction output length does not "
                    "match input record count."
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


        # =============================================
        # 8.4 BUILD API RESULTS
        # =============================================

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


            if prediction_code not in {
                0,
                1,
            }:

                raise RuntimeError(
                    "Model threshold policy returned an "
                    f"unsupported prediction code: "
                    f"{prediction_code}"
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
    # 9. OPERATIONAL MODEL METADATA
    # =====================================================

    def metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return operational metadata for the currently loaded
        production model.

        Used by:

            GET /ready
            GET /v1/model
            structured logs
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