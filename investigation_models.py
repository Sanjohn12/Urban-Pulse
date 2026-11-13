"""Strict contracts for supervised investigations; no model fitting or I/O."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

Text = Annotated[str, Field(min_length=1, max_length=4000)]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,79}$")]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class InvestigationStatus(str, Enum):
    DETECTED = "detected"
    EVIDENCE_COLLECTED = "evidence_collected"
    PROPOSED = "proposed"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EVALUATED = "evaluated"
    ACCEPTED_OR_REJECTED = "accepted_or_rejected"
    RECORDED = "recorded"


class BaselineIdentity(Contract):
    model_id: Text
    model_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    oof_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    eligible_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    metadata_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_sha256: Text
    fold_manifest_sha256: Text
    n_cells: int = Field(gt=0)
    features: list[Identifier]
    model_parameters: dict[str, Any]
    reference_metrics: dict[str, float]
    provenance: dict[str, Any]
    evidence_scope: Literal["explored_saved_spatial_oof"] = "explored_saved_spatial_oof"


class LocalToolCall(Contract):
    name: Literal["residual_analysis", "location_context", "model_explanation", "detect_patterns"]
    query: Literal[None] = None
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class SearchToolCall(Contract):
    name: Literal["literature_search", "data_discovery"]
    query: Annotated[str, Field(min_length=3, max_length=240)]
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class ToolCall(RootModel[Annotated[
    LocalToolCall | SearchToolCall, Field(discriminator="name"),
]]):
    """Structurally separates query-free local tools from bounded searches."""
    def __init__(self, root=None, **data):
        super().__init__(root=root if root is not None else data)

    @property
    def name(self):
        return self.root.name

    @property
    def query(self):
        return self.root.query

    @property
    def limit(self):
        return self.root.limit


class EvidenceReference(Contract):
    evidence_id: Text
    fields: list[Text] = Field(min_length=1, max_length=12)


class LiteratureReference(Contract):
    source_id: Text
    supporting_excerpt: Annotated[str, Field(min_length=8, max_length=350)]
    proposed_relevance: Text


class FeatureDefinition(Contract):
    name: Identifier
    expression: Annotated[str, Field(min_length=1, max_length=400)]
    inputs: list[Identifier] = Field(min_length=1, max_length=12)
    expected_relationship: Text
    rationale: Text


class DataNeed(Contract):
    variable: Identifier
    description: Text
    reason: Text
    spatial_resolution: Text
    temporal_alignment: Text
    license_requirement: Literal["must_be_verified"] = "must_be_verified"


class EvaluationCriteria(Contract):
    daily_r2: Literal["increase"] = "increase"
    daily_rmse: Literal["decrease"] = "decrease"
    daily_mae: Literal["decrease"] = "decrease"
    hourly_profiles: Literal["evaluate_per_hour_and_pooled"] = "evaluate_per_hour_and_pooled"
    independent_spatial_validation: Literal[True] = True
    use_final_holdout_for_selection: Literal[False] = False
    causal_claims: Literal[False] = False


class HypothesisProposal(Contract):
    title: Text
    hypothesis: Text
    mechanism: Text
    targets: Literal["daily_and_hourly_activity"] = "daily_and_hourly_activity"
    method: Literal["statistical_refinement", "theory_feature", "additional_data"]
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=10)
    literature: list[LiteratureReference] = Field(min_length=1, max_length=8)
    features: list[FeatureDefinition] = Field(default_factory=list, max_length=8)
    data_needs: list[DataNeed] = Field(default_factory=list, max_length=8)
    expected_relationships: list[Text] = Field(min_length=1, max_length=8)
    uncertainties: list[Text] = Field(min_length=1, max_length=8)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)

    @model_validator(mode="after")
    def require_testable_change(self):
        if self.method == "theory_feature" and not self.features:
            raise ValueError("A theory-feature proposal needs a symbolic feature definition")
        if self.method == "additional_data" and not self.data_needs:
            raise ValueError("Additional-data proposals need explicit data requirements")
        return self


class ToolPlannerTurn(Contract):
    action: Literal["tool"]
    tool: ToolCall


class ProposalPlannerTurn(Contract):
    action: Literal["propose"]
    proposal: HypothesisProposal


class M2HypothesisProposal(HypothesisProposal):
    """Phase-5 compact evolution accepts a symbolic theory feature only."""
    method: Literal["theory_feature"] = "theory_feature"
    features: list[FeatureDefinition] = Field(min_length=1, max_length=8)


class M2ProposalPlannerTurn(Contract):
    action: Literal["propose"]
    proposal: M2HypothesisProposal


class PlannerTurn(RootModel[Annotated[
    ToolPlannerTurn | ProposalPlannerTurn, Field(discriminator="action"),
]]):
    """The JSON schema itself requires the matching payload, not just Python."""
    @property
    def action(self):
        return self.root.action

    @property
    def tool(self):
        return self.root.tool if isinstance(self.root, ToolPlannerTurn) else None

    @property
    def proposal(self):
        return self.root.proposal if isinstance(self.root, ProposalPlannerTurn) else None


class ExperimentConfiguration(Contract):
    experiment_id: Text
    parent_model_id: Text
    candidate_model_reference: Text
    method: Literal["statistical_refinement", "theory_feature", "additional_data"]
    model_parameters: dict[str, Any]
    base_features: list[Identifier]
    proposed_features: list[FeatureDefinition]
    data_needs: list[DataNeed]
    dataset_sha256: Text
    cohort_sha256: Text
    fold_manifest_sha256: Text
    evidence_digest: Text
    software_versions: dict[str, str]
    implementation_sha256: dict[str, str]
    proposed_service_interface: Text
    targets: Literal["daily_and_hourly_activity"] = "daily_and_hourly_activity"
    target_transform: Literal["log1p"] = "log1p"
    inverse_transform: Literal["expm1"] = "expm1"
    validation_requirement: Literal[
        "new_untouched_spatial_holdout_or_outer_train_only_nested_selection"
    ] = "new_untouched_spatial_holdout_or_outer_train_only_nested_selection"
    selection_evidence_scope: Literal["explored_saved_spatial_oof"] = "explored_saved_spatial_oof"
    independent_validation_ready: Literal[False] = False
    execution_enabled: Literal[False] = False


class ApprovalRequest(Contract):
    request_id: Text
    action: Literal["experiment", "external_data_acquisition", "model_promotion"]
    binding_digest: Text
    requested_at: Text
    decision: Literal["pending", "approved", "denied"] = "pending"
    reviewer: str | None = None
    rationale: str | None = None
    decided_at: str | None = None


class Metrics(Contract):
    r2: float = Field(le=1, strict=True)
    rmse: float = Field(ge=0, strict=True)
    mae: float = Field(ge=0, strict=True)
    n: int = Field(gt=1, strict=True)


class ExecutionPlan(Contract):
    """Concrete host-reviewed execution contract, separate from a proposal."""
    execution_id: Text
    environment: Literal["synthetic_fixture", "research"]
    validation_mode: Literal["exploratory_spatial_cv", "protected_holdout"]
    dataset_path: Text
    dataset_sha256: Text
    cohort_sha256: Text
    split_manifest_sha256: Text
    development_grid_ids: list[Text]
    holdout_grid_ids: list[Text]
    model_parameters: dict[str, Any]
    calibration_inner_splits: int = Field(ge=2, le=5, strict=True)
    minimum_separation_m: float = Field(ge=2500)
    spatial_crs: Literal["EPSG:32644"] = "EPSG:32644"
    holdout_attestation: dict[str, str] = Field(default_factory=dict)
    estimated_rf_fits: int = Field(gt=0)
    implementation_sha256: dict[str, str]
    software_versions: dict[str, str]
    output_policy: Literal["new_experiment_runs_only_no_model_deployment"] = "new_experiment_runs_only_no_model_deployment"


class ValidationReceipt(Contract):
    """Imported measurements from trusted evaluator code, never from the planner."""
    experiment_id: Text
    binding_digest: Text
    configuration_digest: Text
    parent_model_sha256: Text
    dataset_sha256: Text
    cohort_sha256: Text
    fold_manifest_sha256: Text
    candidate_model_reference: Text
    daily: Metrics
    hourly: Metrics | None
    evaluator: Text
    limitations: list[Text] = Field(min_length=1)
    scope: Literal["exploratory_only", "protected_spatial_holdout"] = "exploratory_only"
    independent_spatial_validation: bool = False
    execution_evidence: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_independence(self):
        if self.independent_spatial_validation and (
            self.scope != "protected_spatial_holdout" or self.execution_evidence is None
        ):
            raise ValueError("Independent results need a protected executor evidence record")
        return self


class InvestigationState(Contract):
    schema_version: Literal["1.0"] = "1.0"
    investigation_id: Text
    grid_id: Text
    query: Text
    status: InvestigationStatus = InvestigationStatus.DETECTED
    baseline: BaselineIdentity
    location: dict[str, Any] = Field(default_factory=dict)
    daily_residual_evidence: dict[str, Any] | None = None
    hourly_residual_evidence: dict[str, Any] | None = None
    evidence: dict[str, dict[str, Any]] = Field(default_factory=dict)
    literature_sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    proposal: HypothesisProposal | None = None
    proposal_origin: Literal["none", "validated_ollama", "injected_test_or_custom"] = "none"
    proposal_revision: int = Field(default=0, ge=0)
    experiment_configuration: ExperimentConfiguration | None = None
    execution_plan: ExecutionPlan | None = None
    execution_status: Literal["not_configured", "configured", "running", "completed", "failed"] = "not_configured"
    validation_results: ValidationReceipt | None = None
    validation_artifact: dict[str, Any] | None = None
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    review_status: Literal["not_reviewed", "approved_for_action", "changes_requested", "accepted", "rejected", "deferred"] = "not_reviewed"
    human_reviews: list[dict[str, str]] = Field(default_factory=list)
    scientific_memory_reference: dict[str, str] | None = None
    planner_backend: Literal["ollama", "injected_test_or_custom"]
    planner_status: Literal["not_called", "valid_response", "invalid_response", "unavailable"] = "not_called"
    planner_diagnostics: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    status_history: list[dict[str, str]] = Field(default_factory=list)
    protected_grid_ids: list[str] = Field(default_factory=list)
