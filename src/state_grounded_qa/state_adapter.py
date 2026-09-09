"""Base interface for mapping source-specific records Phi(z_t) to runtime state S_t."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Mapping

from .schemas import RuntimeState


_NON_IDENTIFIER = re.compile(r"[^a-z0-9_.:-]+")


class StateAdapter(ABC):
    """Adapter contract for request-level state construction.

    Offline reference labels must be returned separately by dataset preparation
    code and must never be inserted into the runtime state.
    """

    source_name = "base"

    def __init__(self, missing_value: Any = None) -> None:
        self.missing_value = missing_value

    @abstractmethod
    def build_state(self, raw_record: Mapping[str, Any]) -> RuntimeState:
        """Apply Phi to a raw record and return only information available at t."""

    def normalize_identifier(self, value: Any) -> str | None:
        """Normalize environment identifiers without inventing missing values."""

        if self.is_missing(value):
            return None
        normalized = str(value).strip().lower().replace(" ", "_")
        normalized = _NON_IDENTIFIER.sub("_", normalized).strip("_")
        return normalized or None

    def normalize_identifier_list(self, values: Any) -> list[str]:
        """Normalize and de-duplicate an identifier sequence while preserving order."""

        if self.is_missing(values):
            return []
        if isinstance(values, (str, bytes)):
            values = [values]
        result: list[str] = []
        for value in values:
            normalized = self.normalize_identifier(value)
            if normalized is not None and normalized not in result:
                result.append(normalized)
        return result

    def is_missing(self, value: Any) -> bool:
        """Return whether a source field is missing under the configured policy."""

        return value is None or value == "" or value == self.missing_value

    @staticmethod
    def get_path(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
        """Resolve a dot-separated mapping path."""

        if path in record:
            return record[path]
        current: Any = record
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    @staticmethod
    def metadata_without_references(raw_record: Mapping[str, Any]) -> dict[str, Any]:
        """Return metadata after excluding common offline-label containers."""

        blocked = {"reference", "references", "reference_labels", "ground_truth"}
        metadata = raw_record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            return {}
        return {key: value for key, value in metadata.items() if key not in blocked}
