# Setup Notes

## Overall
Setup was straightforward with no blocking issues.

## Notes

**Weaviate container**
Already running from the drill. Verified with:
```bash
docker ps
curl http://localhost:8080/v1/.well-known/ready
```
Both passed on first try.

**Embeddings**
`all-MiniLM-L6-v2` was already cached from the drill — no download needed.
Embedding ~1,200 corpus rows took ~49 seconds on local hardware (19 batches at 64 texts each).

**One thing to watch**
If you re-run `run_eval.py` without skipping `create_schema` and `index_corpus`,
it re-ingests the full corpus. Comment those two lines out after the first run
to save ~1 minute per iteration.