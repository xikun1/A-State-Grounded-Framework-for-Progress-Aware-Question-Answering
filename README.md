# State-Grounded QA for Interactive 3D Environments

This repository contains the state-grounded, progress-aware question-answering implementation for interactive 3D learning environments. It includes the core pipeline, source adapters, deterministic verification, evaluation procedures, robustness-input generation, schemas, tests, and the author-verified UE5 request collection. It does not contain paper figures or aggregate result files.

## Core pipeline

```text
raw records -> state adapter -> runtime state S_t
-> progress/legal actions -> state-conditioned retrieval
-> structured generation -> deterministic verification
-> one retry or conservative fallback -> final response
```

The retry reuses the same runtime state and selected evidence. The environment remains authoritative; the pipeline does not execute actions or mutate task state.

The progress verifier checks active-route membership, confirmed completion, prerequisite/order constraints, completion-status compatibility, required active suffix steps, and duplicate steps. It does not require every representation field to be byte-for-byte equal to the runtime state.

## Repository structure

```text
src/state_grounded_qa/                  Core method, adapters, retrieval, verification
configs/                                Model, retrieval, and UE5 adapter configuration
data/schemas/                           Runtime, task-graph, evidence, response schemas
data/ue5/                               UE5 export schemas and format guidance
UE5_24_sequences_480_requests/          24-sequence, 480-request UE5 research input collection
evaluation/                             Metrics, independent reference checks, robustness transforms
scripts/                                Data preparation and evaluation entry points
tests/                                  Unit tests and lightweight fixtures
```

Evaluation outputs are written to user-selected paths outside the repository, for example `../evaluation_outputs/`.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

The sentence-transformer backend loads `BAAI/bge-m3` only when embedding is requested. Retrieval applies state applicability filtering first, then uses FAISS `IndexFlatIP` exact search over L2-normalized float32 vectors, so inner products are cosine similarities. Top-k is 5 and reranking is disabled.

Evidence chunk lengths are counted independently with `Qwen/Qwen2.5-7B-Instruct` using `add_special_tokens=False` and are limited to 256 tokens. The tokenizer is loaded lazily and `StateGroundedPipeline(..., token_counter=...)` accepts an injected counter for offline tests.

## Data and adapters

MSQA and EAI–VirtualHome source data are not redistributed. Use the corresponding preparation scripts after obtaining the source records under their applicable terms:

```bash
python scripts/prepare_msqa.py --input path/to/msqa.jsonl --output ../evaluation_outputs/msqa.jsonl
python scripts/prepare_eai_virtualhome.py --input path/to/eai_requests.jsonl --output ../evaluation_outputs/eai.jsonl
python scripts/normalize_ue5_replay.py --input path/to/ue5_export.jsonl --output ../evaluation_outputs/ue5_normalized.jsonl
```

The included `UE5_24_sequences_480_requests/` collection contains normalized request records, reference labels, manifest metadata, per-sequence files, and six task graphs. `tests/fixtures/ue5/` is separate lightweight parser-test input.

## Progress and action evaluation

Runtime-state fields (`current_step`, `remaining_steps`) and structured-response fields (`current_step_id`, `remaining_step_ids`) are normalized before scoring. The evaluator keeps `state`, `initial_candidate`, `final_verified`, and legacy `prediction` stages independent. Structured responses without `completed_steps` or `legal_actions` receive unavailable (`null`) F1 values rather than being treated as empty sets.

```bash
# Table 5: Progress-state Construction
python scripts/run_progress_evaluation.py --input INPUT.jsonl --output ../evaluation_outputs/state_metrics.json --track state --stage state

# Table 6: Progress and Action Grounding
python scripts/run_progress_evaluation.py --input INPUT.jsonl --output ../evaluation_outputs/final_metrics.json --track progress_action --stage final_verified
```

Next-action Accuracy counts null actions as incorrect, including when the acceptable-action reference set is empty. Action Validity and Illegal-action rates use non-null proposed actions as their denominator. `non_null_action_recommendation_rate` reports coverage. Precondition/Order Violation Rate also uses non-null actions; missing independent reference constraints are reported as unevaluable rather than false. Duplicate-step scoring is unchanged.

## Verifier evaluation

All verifier procedures use the existing pipeline, one retry at most, and conservative fallback where needed. For offline evaluation, `retrieved_evidence` is the preferred already-selected generation-time evidence collection; `evidence` remains compatible as the legacy selected collection. The records are serialized and verified in their supplied order without re-filtering, embedding, ranking, or top-k truncation, and the same collection is used for the initial candidate, retry, and fallback. Independent reference information must be supplied for the dimensions being checked. Legal actions are resolved in this order: `reference_legal_actions`, `reference_labels.reference_legal_actions`, `reference.legal_actions`, then `reference_runtime_state.action.available_actions`; an explicit empty list is authoritative, and a missing set is unevaluable rather than copied from `runtime_state`. The same reference context is rechecked against the actual initial, retry, and final responses. Supplied final labels are retained only as `supplied_final_reference_*` cross-check fields; formal final validity and residual labels come from the recomputed final response. Full-sample residual and valid non-fallback rates are `null` when any final reference check is unevaluable, with evaluability counts retained.

```bash
# Table 10: Controlled Verification, initial detection
python scripts/run_verifier_evaluation.py --procedure controlled --input path/to/accepted_candidates.jsonl --output ../evaluation_outputs/controlled_metrics.json

# Table 10: Controlled Verification, full residual procedure
python scripts/run_verifier_evaluation.py --procedure controlled-full --input path/to/candidates_with_retries.jsonl --evaluation-set ../evaluation_outputs/controlled_outcomes.jsonl --output ../evaluation_outputs/controlled_full_metrics.json

# Table 11: Verification on Naturally Generated Candidates
python scripts/run_verifier_evaluation.py --procedure natural --input path/to/candidates.jsonl --evaluation-set ../evaluation_outputs/natural_outcomes.jsonl --output ../evaluation_outputs/natural_metrics.json

# Table 11: separate EAI-VirtualHome and UE5 records plus CSV summaries
python scripts/run_table11_evaluation.py --eai-input path/to/eai_candidates.jsonl --ue5-input path/to/ue5_candidates.jsonl --output-dir ../evaluation_outputs/table11

# Table 12: Verification Acceptance and Recovery
python scripts/run_verifier_evaluation.py --procedure full --input path/to/request_candidates.jsonl --evaluation-set ../evaluation_outputs/full_outcomes.jsonl --output ../evaluation_outputs/full_metrics.json

# Table 11 action/reference consistency audit (optional Table 6/14 outputs are external)
python scripts/audit_table11.py --eai-records ../evaluation_outputs/table11/table11_eai_records.csv --ue5-records ../evaluation_outputs/table11/table11_ue5_records.csv --ue5-replay UE5_24_sequences_480_requests/replay_requests.jsonl --ue5-labels UE5_24_sequences_480_requests/reference_labels.jsonl --output ../evaluation_outputs/table11/table11_consistency_audit.json
```

`natural` first runs the independent reference check on the original LLM candidate, then runs online verification, a real retry if rejected, fallback if needed, and an independent final reference check. It reports natural violation rate before verification, request-level detection recall, false rejection, retry, recovery, fallback, residual violation, and valid non-fallback coverage, with numerators, denominators, and reference-evaluation completeness. Controlled experiments retain their seven-class detection and residual aggregation separately.

The procedure-to-table mapping is explicit: `controlled` and `controlled-full` are Table 10, `natural` is Table 11, and `full` is Table 12. Table 11 records preserve initial, retry, and final verifier results separately from independent reference validity, including supplied-label agreement fields. Its action audit indexes records by dataset, request ID, run ID, and explicitly supplied evaluation stage. Only complete identical identities are strictly compared; missing metadata, different runs, different stages, and duplicate identities are reported rather than inferred or overwritten. The audit checks that final illegal-action requests are included in the final residual-reference set when the same request-level records are supplied. The project does not contain the original EAI-VirtualHome or LLM candidate records; provide those records through the command-line input paths.

## Environment-record robustness inputs

`stale` and `distractor` remain available. `record-anomaly` generates deterministic inputs with unchanged request IDs and reference labels under four conditions: `clean`, `incomplete`, `contradictory`, and `delayed`.

```bash
python scripts/run_robustness.py --mode record-anomaly --condition clean --input path/to/records.jsonl --output ../evaluation_outputs/records_clean.jsonl
python scripts/run_robustness.py --mode record-anomaly --condition incomplete --field-path runtime_state.task.current_step --missing-rule null --input path/to/records.jsonl --output ../evaluation_outputs/records_incomplete.jsonl
python scripts/run_robustness.py --mode record-anomaly --condition contradictory --field-path runtime_state.task.current_step --conflict-value '"conflicting_step"' --input path/to/records.jsonl --output ../evaluation_outputs/records_contradictory.jsonl
python scripts/run_robustness.py --mode record-anomaly --condition delayed --field-path runtime_state --delay-steps 1 --seed 0 --input path/to/records.jsonl --output ../evaluation_outputs/records_delayed.jsonl
```

Multiple `--field-path` values may be supplied. Selection is deterministic for a fixed `--seed`. The anomaly metadata records the condition, selected path, delay or missing rule, and seed.

## UE5 and computational-cost evaluation

Timing records contain request-level `latency_s`, optional `stage_latency_s`, `tokens`, `evaluation_protocol`, `evaluation_subset`, `timing_scope`, and `execution_config`. `latency_s` is the supplied endpoint-to-endpoint request value. Stage means and the mean difference between request latency and stage-sum are reported separately; stage sums never replace request latency. Use semantic protocol names rather than paper table numbers: `ue5_offline_replay_evaluation` for Table 14, and `end_to_end_computational_cost_evaluation` for the Table 18/19 computational-cost records. Table 19 is read from `stage_wise_latency_s` in the selected computational-cost summary.

```bash
# Table 14: UE5 Offline Replay Performance
python scripts/profile_efficiency.py --input UE5_TIMING.jsonl --output ../evaluation_outputs/ue5_offline_replay_metrics.json --protocol ue5_offline_replay_evaluation

# Table 18/19: End-to-end Computational Cost and Stage-wise Latency Breakdown
python scripts/profile_efficiency.py --input COST_TIMING.jsonl --output ../evaluation_outputs/computational_cost_metrics.json --protocol end_to_end_computational_cost_evaluation

# Compare explicitly labelled protocols without mixing them
python scripts/profile_efficiency.py --input TIMING.jsonl --output ../evaluation_outputs/protocol_comparison.json --compare-protocols
```

Each protocol is aggregated independently with sample count, mean latency, P95 latency, token average, stage averages, timing scope, subset, execution configuration, and consistency diagnostics. The program does not infer a protocol from a paper table number; request-level timing records must supply the semantic protocol name.

## Tests

```bash
python -m pytest -q -p no:cacheprovider
```

Tests use lightweight fixtures. Temporary evaluation outputs should be written outside the repository. Python caches, virtual environments, logs, generated outputs, and image/PDF artifacts are ignored by `.gitignore`.
