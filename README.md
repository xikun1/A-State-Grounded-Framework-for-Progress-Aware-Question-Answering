# State-Grounded QA for Interactive 3D Environments

This repository contains the implementation and evaluation resources accompanying the state-grounded, progress-aware question-answering framework for interactive 3D environments.

## Overview

The code exposes the paper's common request-level state contract, task-progress and legal-action derivation, state-conditioned retrieval, structured generation interface, deterministic verification, one-retry policy, and conservative fallback. Dataset adapters and evaluation utilities keep runtime inputs separate from offline reference labels.

## Method pipeline

```text
raw records
→ state adapter
→ runtime state S_t
→ task progress and legal actions
→ state-conditioned evidence retrieval
→ structured generation
→ deterministic verification
→ one retry or conservative fallback
→ final response
```

The retry reuses the same `S_t` and selected evidence `E_t`; retrieval is not repeated. The environment remains authoritative and the pipeline does not execute actions or mutate state.

## Repository structure

```text
src/state_grounded_qa/   Core method and source adapters
configs/                 Model, retrieval, and UE5 adapter configuration
data/schemas/            Runtime, task-graph, evidence, and response schemas
data/ue5/                UE5 manifest/request schemas and data guidance
UE5_24_sequences_480_requests/  Author-verified UE5 replay records, labels, and task graphs
evaluation/              Metrics and controlled robustness transformations
scripts/                 Dataset preparation and evaluation entry points
plotting/                Matplotlib scripts for reported figures
results/reported_metrics Reported aggregate metrics for tables/figures
tests/                   Unit tests and software-test fixtures
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
# Activate the environment for your shell, then:
pip install -r requirements.txt
pip install -e .
```

The sentence-transformer backend loads `BAAI/bge-m3` only when embedding is requested; importing the package does not download model weights.

## Data

### MSQA

Obtain the MSQA data through the official [MSR3D project](https://msr3d.github.io/) and [MSR3D repository](https://github.com/MSR3D/MSR3D). The data are not bundled here.

### EAI–VirtualHome

Obtain the trajectory and task resources from the [Embodied Agent Interface repository](https://github.com/embodied-agent-interface/embodied-agent-interface) or its [Hugging Face dataset page](https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface). Runtime construction uses only the trajectory prefix before the current request.

### UE5

The author-verified UE5 replay collection used for the cross-domain offline replay evaluation is included under `UE5_24_sequences_480_requests/`. It contains 24 sequences and 480 request-level records together with the manifest, reference labels, per-sequence files, and task graphs. These files are the normalized research records used by the evaluation workflow; original UE5 project files, 3D assets, textures, audio, Marketplace assets, and executables are not distributed. Software-test records under `tests/fixtures/ue5/` are separate fixtures used only for parser and schema tests and are not part of the experimental collection.

## Running core components

All commands are run from the repository root. Inputs are user-supplied files obtained under the applicable data terms.

```bash
python scripts/prepare_msqa.py --input path/to/msqa.jsonl --output outputs/msqa.jsonl
python scripts/prepare_eai_virtualhome.py --input path/to/eai_requests.jsonl --output outputs/eai.jsonl
python scripts/normalize_ue5_replay.py --input path/to/ue5_export.jsonl --output outputs/ue5_normalized.jsonl
python scripts/run_scene_evaluation.py --input path/to/scene_predictions.jsonl --output outputs/scene_metrics.json
python scripts/run_progress_evaluation.py --track progress_action --input path/to/progress_predictions.jsonl --output outputs/progress_metrics.json
python scripts/run_verifier_evaluation.py --input path/to/accepted_candidates.jsonl --output outputs/verifier_metrics.json
python scripts/run_robustness.py --mode stale --delta 2 --input path/to/sequences.jsonl --output outputs/stale_delta2.jsonl
python scripts/profile_efficiency.py --input path/to/timing_records.jsonl --output outputs/efficiency.json
```

The verifier evaluation creates seven controlled transformations per supplied accepted candidate. The number of candidates is determined by the input; no fixed-size evaluation collection is bundled.

## Plotting

The plotting utilities read only `results/reported_metrics/` and write PNG/PDF files to `results/figures/`.

```bash
python plotting/plot_fig2.py
python plotting/plot_fig3.py
python plotting/plot_fig5.py
python plotting/plot_fig6.py
```

## Configuration

`configs/model.yaml` records Qwen2.5-7B-Instruct, FP16, deterministic decoding, the 8,192-token context policy, and one maximum retry. `configs/retrieval.yaml` records BAAI/bge-m3, cosine similarity, Top-k 5, chunks of at most 256 tokens, and no reranker.

The controlled profiling configuration reported in the paper used an NVIDIA RTX 3090 24 GB, a 16-core CPU, 64 GB RAM, and batch size 1. The implementation does not require identical hardware.

## Data availability

Third-party datasets remain subject to their respective release and license terms and are not redistributed in this repository. The author-verified normalized UE5 replay records used in the reported offline replay evaluation are included under `UE5_24_sequences_480_requests/`. UE5 project files, 3D assets, textures, audio, Marketplace assets, and executables remain outside the scope of this research-code release.
