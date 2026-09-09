# Data access and schemas

This directory contains schemas and access instructions, not redistributed third-party records.

## MSQA

Acquire MSQA from the official MSR3D sources:

- https://msr3d.github.io/
- https://github.com/MSR3D/MSR3D

Use `scripts/prepare_msqa.py` to map source records to the unified scene-state representation. Task-progress fields remain non-applicable when the source record does not define them.

## EAI–VirtualHome

Acquire EAI–VirtualHome resources from:

- https://github.com/embodied-agent-interface/embodied-agent-interface
- https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface

Use `scripts/prepare_eai_virtualhome.py` on request-level records containing a task specification and a trajectory prefix boundary. Future actions, future events, and final outcomes must not be exposed to the runtime adapter.

## UE5 exports

See `data/ue5/README.md`. The repository accepts exported JSON, JSONL, and CSV records and does not depend on an Unreal Engine runtime.
