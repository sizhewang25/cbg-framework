# CBG End-to-End Setting Evaluation

Refactor the CBG evaluation runner so each `PipelineSpec` is evaluated as one
end-to-end setting: data loading, data preparation, model cache lookup or
fitting, pipeline construction, and probe evaluation.

