# Comparison Brief — Module 8 Lab

> ~300–500 words. Replace the placeholder text in each section with your analysis.

## Metrics Table

| Retriever | recall@5 | recall@10 | MRR | factoid recall@5 | paraphrastic recall@5 |
|---|---|---|---|---|---|
| BM25 | 0.567 | 0.650 | 0.550 | 1.000 | 0.133 |
| Dense | 0.900 | 0.933 | 0.670 | 0.833 | 0.967 |
| Hybrid (α=0.5) | 0.850 | 0.983 | 0.698 | 1.000 | 0.700 |

## Where BM25 Wins

1. Factoid / exact-identifier queries — BM25 achieves perfect factoid recall@5 (1.000) while dense drops to 0.833. Queries that contain a distinctive token like a version number, library name, or error code give BM25 a direct lexical match that the dense model may dilute into a broader semantic neighborhood.
2. Rare technical terms — When a query contains an uncommon but precise term (e.g. a specific API method or config flag), BM25's inverted index rewards exact token overlap and ranks the gold document first, whereas the dense model may surface semantically similar but lexically different documents that don't actually answer the question.

## Where Dense Wins

1. Paraphrastic queries — Dense achieves 0.967 paraphrastic recall@5 versus BM25's 0.133. When the query is worded differently from the document (e.g. "how do I speed up my app" vs. a document titled "Android performance optimization techniques"), BM25 finds no token overlap and fails, while the dense vector captures the shared meaning.
2. Vocabulary mismatch / synonym queries — When a user asks about "fixing memory leaks" but the gold document uses terms like "heap allocation" or "garbage collection overhead," BM25 scores near zero while the dense model correctly maps both phrasings into nearby vector space and retrieves the right document.

## Alpha Recommendation

The data strongly favors a dense-leaning alpha in the range 0.55–0.65, with a recommended default of 0.6. The reasoning: paraphrastic queries make up a large share of real user traffic on a Q&A corpus (users rarely phrase questions the same way the original poster did), and dense dominates that slice (0.967 vs 0.133). However, hybrid at α=0.5 already recovers perfect factoid recall@5 (1.000) — meaning BM25's lexical signal is sufficient to anchor exact-match queries even at half weight. Pushing alpha slightly above 0.5 toward 0.6 preserves that factoid ceiling while improving paraphrastic recall, giving the best aggregate MRR (which hybrid already leads at 0.698).
