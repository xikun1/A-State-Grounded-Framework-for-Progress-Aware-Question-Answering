# Repository contents and release audit

This repository contains:

- the state-grounded method implementation, including retrieval, serialization, verification, retry, and fallback;
- MSQA, EAI–VirtualHome, and exported-log UE5 adapters;
- dataset preparation and normalization scripts;
- scene, state/progress, verifier, robustness, and efficiency evaluation code;
- reported aggregate metrics transcribed for table and figure generation;
- Matplotlib plotting utilities;
- UE5 normalized-record schemas and adapter configuration;
- the author-verified UE5 offline replay collection with 24 sequences and 480 request-level records, reference labels, manifest metadata, and task graphs.

Release audit:

1. `UE5_24_sequences_480_requests/` contains the author-verified normalized UE5 replay collection used for the reported cross-domain offline replay evaluation: 24 sequences and 480 request-level records, with manifest metadata, reference labels, per-sequence files, and task graphs.
2. `tests/fixtures/ue5/` is explicitly separated from the experimental collection and is used only as software-test input.
3. Reported aggregate metrics are stored under `results/reported_metrics/`; UE5 request-level replay records and reference labels are stored separately under `UE5_24_sequences_480_requests/`.
4. MSQA and EAI–VirtualHome source data are not redistributed; only official access instructions are provided.
5. Every Python file contains executable code or package documentation; no Python file is empty.
6. Core modules contain working logic rather than TODO-only bodies.
7. README commands correspond to scripts present in `scripts/` and `plotting/`.
8. Plotting scripts read the reported-metrics CSV files and write figures under `results/figures/`.
9. Runtime dependencies correspond to imports used by the implementation.
