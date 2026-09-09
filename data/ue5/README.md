# UE5 exported-record interface

This directory defines the format expected for externally exported UE5 interaction records:

- `replay_manifest.schema.json` documents sequence provenance and source-file hashes.
- `request_record.schema.json` defines normalized request-level records.
- `task_graph.schema.json` points to the common task-graph schema.
- `../../configs/ue5_adapter.yaml` maps source fields to the common state.
- `../../scripts/normalize_ue5_replay.py` converts JSON, JSONL, or CSV exports.

Normalized requests contain `sequence_id`, `request_id`, `timestamp`, `scene`, `task`, `action`, `event`, `dialogue`, `runtime_state`, and `question`.

The author-verified normalized UE5 replay collection used for the reported offline replay evaluation is distributed at `../../UE5_24_sequences_480_requests/`. It contains 24 sequences and 480 request-level experimental records, together with manifest metadata, reference labels, per-sequence files, and task graphs. The records under `tests/fixtures/ue5/` are separate software-test fixtures used only to test parsing and schema behavior.
