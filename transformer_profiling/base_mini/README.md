# BERT-mini base profiling TSVs

Each experiment TSV has seven rows: six `interval_delta` rows derived by subtracting adjacent cumulative gem5 dumps, plus one `final_total` row from dump 6.

The six intervals are `MHA`, `Projection`, `non_GEMM_after_projection`, `FF1`, `FF2`, and `non_GEMM_after_ff2`.

Ratio columns (`ipc`, `cpi`, cache miss rates, branch misprediction rate, and `btb_hit_ratio`) are intentionally blank for interval rows and populated only for `final_total`.

Use `base_mini_all_experiments.tsv` for one combined table, or the `E*.tsv` files for per-experiment plotting. See `metric_sources.tsv` for source stat names.
