# Evals

Evaluation scripts for the application live here. These scripts can depend on
external services such as LangSmith, model providers, PostgreSQL, Redis, Milvus,
and Neo4j.

Run the LangSmith RAG evaluation from the repository root:

```bash
uv run python evals/langsmith_eval.py
```

Set `LANGSMITH_EVAL_DATASET` to override the default dataset name (`RAG`).
