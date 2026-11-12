# Repository contents

This release contains the method implementation, source adapters, data schemas, evaluation code, tests, and the author-verified UE5 normalized research input collection.

## Included

- `src/state_grounded_qa/`: runtime state contract, task progress, retrieval, generation interface, verification, retry, and fallback.
- `configs/`: model, retrieval, and UE5 adapter configuration.
- `data/schemas/` and `data/ue5/`: normalized-record schemas and format guidance.
- `evaluation/`: progress/action, verifier, efficiency, independent-reference, and robustness-input logic.
- `scripts/`: data preparation, Table 5/6 progress-action evaluation, Table 10/11/12 verifier evaluation, Table 11 aggregation/audit, UE5 Table 14 and Table 18/19 timing evaluation, and robustness-input commands.
- `tests/`: unit tests and lightweight fixtures.
- `UE5_24_sequences_480_requests/`: 24 sequences and 480 request-level normalized UE5 records with reference labels, manifest metadata, and six task graphs.

## Release rules

1. Paper figures, aggregate paper-result tables, and generated evaluation outputs are not included.
2. Timing, verifier, progress, and robustness commands write outputs only to paths supplied by the user.
3. MSQA and EAI–VirtualHome source records remain external; their preparation scripts and access guidance are included.
4. `tests/fixtures/ue5/` is software-test input and is separate from the UE5 research collection.
5. `.gitignore` excludes Python caches, virtual environments, logs, temporary outputs, and generated image/PDF artifacts.

## Verifier evaluation tracks

`controlled` and `controlled-full` implement the two Table 10 procedures. `natural` and `run_table11_evaluation.py` implement Table 11 natural-candidate evaluation with fixed generation-time evidence (`retrieved_evidence` or legacy `evidence`) and independent initial/retry/final reference checks; a missing reference dimension is unevaluable and never filled from the online runtime state. `full` remains the Table 12 acceptance/recovery procedure. Timing records use semantic protocol names: `ue5_offline_replay_evaluation` for Table 14 and `end_to_end_computational_cost_evaluation` for Table 18/19. All CSV/JSON outputs and consistency audits are written only to user-selected external paths. Table 6/11/14 audit comparisons require complete matching record identities and report missing metadata, different runs/stages, and duplicate identities explicitly.
