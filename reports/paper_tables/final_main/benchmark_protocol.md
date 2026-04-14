# Benchmark Protocol

- Splits: `/LCA_LLM/data/splits_clean/random_stratified_{train,dev,test}.parquet`
- Main retrieval methods: `bm25_only`, `dense_only`, `hybrid_no_rerank`, `hybrid_with_rerank`
- Main regression methods: `top1_factor_lookup`, `topk_factor_mixture`, `lgbm_regressor`
- Main retrieval metrics: `top1_accuracy`, `mrr@10`, `ndcg@10`
- Main regression metrics: `rmse`, `spearman`
- Main uncertainty metrics: `calibration_error`, `retained_risk`, `abstention_gain`, `aurc`
- All model/hyperparameter choices are selected on dev only.
- Test is used only for final reporting.
- Uncertainty threshold and conformal calibration are learned on dev only.
