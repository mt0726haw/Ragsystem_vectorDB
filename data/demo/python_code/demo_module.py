"""Demo module showcasing a small data processing pipeline.

This module is intentionally simple. It exists to demonstrate how the
Python AST chunker emits one chunk per class, function and method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class PipelineResult:
    """Container for pipeline execution results.

    Attributes:
        records: Successfully processed records.
        errors: Tuples of (record, exception) for failed records.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Any] = field(default_factory=list)


class Stage:
    """Base class for all pipeline stages."""

    name: str = "stage"

    def process(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single record. Override in subclasses."""
        raise NotImplementedError


class NormalizeStage(Stage):
    """Normalize sensor values into a [0, 1] range."""

    name = "normalize"

    def process(self, record: Dict[str, Any]) -> Dict[str, Any]:
        value = float(record.get("value", 0.0))
        record["value_normalized"] = max(0.0, min(1.0, value / 100.0))
        return record


class AggregateStage(Stage):
    """Aggregate records by a configurable key (e.g. sensor_id)."""

    name = "aggregate"

    def __init__(self, group_by: str = "sensor_id") -> None:
        self.group_by = group_by
        self._buckets: Dict[Any, List[float]] = {}

    def process(self, record: Dict[str, Any]) -> Dict[str, Any]:
        key = record.get(self.group_by)
        self._buckets.setdefault(key, []).append(float(record.get("value_normalized", 0.0)))
        record["running_avg"] = sum(self._buckets[key]) / len(self._buckets[key])
        return record


class DataPipeline:
    """A minimal multi-stage data processing pipeline."""

    def __init__(self, name: str = "pipeline") -> None:
        self.name = name
        self._stages: List[Stage] = []

    def add_stage(self, stage: Stage) -> "DataPipeline":
        """Append a stage and return self for fluent chaining."""
        self._stages.append(stage)
        return self

    def run(self, records: Iterable[Dict[str, Any]]) -> PipelineResult:
        """Run the pipeline against an iterable of input records."""
        result = PipelineResult()
        for record in records:
            try:
                for stage in self._stages:
                    record = stage.process(record)
                result.records.append(record)
            except Exception as exc:  # noqa: BLE001 - isolate per record
                result.errors.append((record, exc))
        return result


def process_records(records: Iterable[Dict[str, Any]]) -> PipelineResult:
    """High-level helper that builds and runs a default pipeline."""
    pipeline = DataPipeline(name="default")
    pipeline.add_stage(NormalizeStage())
    pipeline.add_stage(AggregateStage(group_by="sensor_id"))
    return pipeline.run(records)
