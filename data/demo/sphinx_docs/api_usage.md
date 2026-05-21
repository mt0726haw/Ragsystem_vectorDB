# Demo API Usage

The demo module exposes a tiny pipeline-like API that processes data
records and produces aggregated results.

## Creating a Pipeline

Use `DataPipeline` from `demo_module` to wire stages together:

```python
from demo_module import DataPipeline, NormalizeStage, AggregateStage

pipeline = DataPipeline(name="demo")
pipeline.add_stage(NormalizeStage())
pipeline.add_stage(AggregateStage(group_by="sensor_id"))
result = pipeline.run(records)
```

## Processing Records

Each pipeline call accepts an iterable of dictionaries. The `process_records`
helper is the recommended entry point for batch processing - it handles
chunking and error isolation per record.

## Error Handling

When a stage raises, the pipeline records the failure and continues with
the next record. The collected errors are available via `result.errors`.

## Performance Tips

- Pre-normalize input data when possible to skip the normalize stage.
- Increase the batch size for the aggregate stage if memory permits.
- Persist intermediate results when running long pipelines.
