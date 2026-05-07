# Final Action Label Emission Inventory

This inventory records every audited path that can emit final `BUY`, `SELL`, or
`HOLD` action labels to a UI surface, report, export artifact, or package API.
It is evidence-only documentation for semantic QA hardening and does not change validation behavior, signal semantics,
model predictions, backtest results, or report metrics.

## Invariant

Final action labels are produced only by the deterministic signal engine after
scoring, cost, slippage, and risk rules are applied. Model and LLM outputs remain
structured features or report-only diagnostics; they do not emit final action
labels and do not decide `BUY`, `SELL`, or `HOLD`.

This evidence does not change validation behavior. The completed-report builder
does not accept raw model prediction sections as final action inputs, and there
is no live order path in any audited label surface. Invariant: only
`src/quant_research/signals/engine.py` assigns final row-level `action` labels.

## Source Of Final Labels

| Path | Function or object | Emission | Notes |
|---|---|---|---|
| `src/quant_research/signals/engine.py` | `DeterministicSignalEngine.generate` | Creates the `action` column with `HOLD` default, then `SELL` and `BUY` masks. | The only production code path that assigns final `BUY`, `SELL`, or `HOLD` labels. |
| `src/quant_research/signals/engine.py` | `require_signal_generation_gate_pass` | Blocks final signal generation when a required validation gate is missing or not `PASS`. | Enforcement hook for production-like final signal paths; low-level research tests may call without requiring a gate. |

## Emission Path Provenance Evidence

Each audited emission path receives labels from the deterministic signal output
DataFrame or from completed-report inputs named `deterministic_signal_outputs`.
No audited UI, report, export, or package API path receives final labels from raw
model output columns or free-form LLM text.

| Emission path | Label source evidence | Model-output isolation evidence |
|---|---|---|
| `src/quant_research/backtest/engine.py` `run_long_only_backtest` | `generated_signals = signal_engine.generate(...)`; `signals["action"] = generated_signals["action"].to_numpy()` | The copied label source is the deterministic engine output, not a model adapter or prediction frame. |
| `src/quant_research/pipeline.py` `PipelineResult.signals` | `signals=backtest.signals` | Pipeline-level outputs pass through `BacktestResult.signals`; model predictions remain in `PipelineResult.predictions`. |
| `app.py` latest deterministic signals table | Reads `result.signals` for the latest date and displays those rows. | The UI table title and source both use deterministic signals; it does not read `result.predictions["action"]`. |
| `app.py` no-model-proxy counts | Reads `action_counts` from deterministic signal evaluation metrics. | Counts are aggregate metrics derived from `result.signals`, not model labels. |
| `src/quant_research/dashboard/beginner.py` `raw_signal` diagnostic | `_latest_action` reads `result.signals` first. | `raw_signal` is explicitly hidden from rendered beginner UI; the defensive `result.predictions["action"]` fallback is documented as a non-emission compatibility fallback only. |
| `src/quant_research/validation/report_generation.py` completed report | `build_completed_validation_backtest_report` accepts `deterministic_signal_outputs`; `_signal_summary` aggregates its `action` column. | The report builder deliberately does not accept raw model prediction sections as final action inputs. |
| `src/quant_research/validation/gate.py` validity report counts | Renders `action_counts` already present in deterministic signal evaluation metrics. | Validity reports summarize deterministic signal metrics; they do not compute labels from model outputs. |
| `src/quant_research/validation/report_renderer.py` structured report renderers | Render payload sections produced by completed-report builders. | Generic renderers only serialize provided report payloads; they do not read prediction frames. |
| `scripts/run_backtest_validation.py` `signals.csv` | Writes `result.backtest.signals` and passes it as `deterministic_signal_outputs`. | Exported CSV and report artifacts derive from deterministic backtest signals. |

## Package API Surfaces

| Path | Function or object | Emission | Notes |
|---|---|---|---|
| `src/quant_research/signals/__init__.py` | `DeterministicSignalEngine`, `SignalEngineConfig` | Exposes the engine that creates final labels. | Public import surface for the only label producer. |
| `src/quant_research/backtest/engine.py` | `run_long_only_backtest` | Copies generated `action` values into `BacktestResult.signals`. | Pass-through from `DeterministicSignalEngine.generate`; it does not invent labels. |
| `src/quant_research/backtest/engine.py` | `BacktestResult.signals` | Returns a DataFrame containing `action`. | Package API output used by UI, reports, exports, and tests. |
| `src/quant_research/backtest/engine.py` | `_select_stateful_targets` | Consumes `SELL` to remove existing holdings and `BUY` to rank candidates. | Long-only backtest selection logic; no live order path. |
| `src/quant_research/backtest/__init__.py` | `BacktestResult`, `run_long_only_backtest` | Exposes backtest API that returns labels in `BacktestResult.signals`. | Public import surface for backtest label pass-through. |
| `src/quant_research/pipeline.py` | `PipelineResult.signals` | Returns the backtest `signals` DataFrame. | Package API pass-through from `BacktestResult.signals`. |
| `src/quant_research/pipeline.py` | `run_research_pipeline` | Assigns `signals=backtest.signals` in `PipelineResult`. | Pipeline-level pass-through; no independent label generation. |
| `src/quant_research/pipeline.py` | `_deterministic_signal_evaluation_metrics`, `_signal_action_counts`, `_signal_action_ratios`, `_flatten_signal_evaluation_metrics` | Emits aggregate counts and ratios for `BUY`, `SELL`, and `HOLD`. | Metric summaries only; these do not create or alter row-level labels. |

## UI Surfaces

| Path | Function or object | Emission | Notes |
|---|---|---|---|
| `app.py` | main Streamlit result rendering, `Latest Deterministic Signals` tab | Displays latest `result.signals` rows including the `action` column. | UI table pass-through from `PipelineResult.signals`. |
| `app.py` | no-model-proxy ablation panel | Displays `buy_count`, `sell_count`, and `hold_count` from deterministic signal evaluation metrics. | Aggregate display only; no row-level label generation. |
| `src/quant_research/dashboard/beginner.py` | `build_beginner_research_dashboard` | Sets `raw_signal` from `_latest_action`. | Dashboard object keeps the raw label for diagnostics while `research_summary["raw_signal_visible"]` is `False`. |
| `src/quant_research/dashboard/beginner.py` | `_latest_action` | Reads the latest label from `result.signals`; falls back to `result.predictions["action"]` only if present, then `HOLD`. | The fallback is defensive for missing data. Production pipeline predictions do not create final labels. |
| `src/quant_research/dashboard/streamlit.py` | `render_beginner_overview` | Does not render `raw_signal`. | Beginner UI renders badges, forecasts, SEC events, and metrics instead of direct action labels. |

## Report Surfaces

| Path | Function or object | Emission | Notes |
|---|---|---|---|
| `src/quant_research/validation/report_schema.py` | `ReportDeterministicSignalSummarySchema` | Documents `action` as `BUY|SELL|HOLD` and sets allowed actions. | Report schema contract for deterministic signal summaries. |
| `src/quant_research/validation/report_schema.py` | `_deterministic_signal_outputs_input_section` | Requires `action` in completed-run report inputs. | Input contract for completed validation/backtest reports. |
| `src/quant_research/validation/report_generation.py` | `build_completed_validation_backtest_report` | Consumes `deterministic_signal_outputs` and creates `deterministic_signal_summary`. | Report builder rejects future-dated signal inputs and does not accept raw model prediction sections. |
| `src/quant_research/validation/report_generation.py` | `_signal_summary` | Emits `action_counts` and `latest_action_counts`. | Aggregates existing labels for report payloads; no new labels are created. |
| `src/quant_research/validation/report_generation.py` | `render_completed_validation_backtest_report` | Renders completed-run report payloads to Markdown or HTML. | Renderer surface for deterministic action aggregates. |
| `src/quant_research/validation/report_generation.py` | `write_completed_validation_backtest_report_artifacts` | Writes canonical report JSON, Markdown, and HTML artifacts. | Exported reports include deterministic action aggregates from `_signal_summary`. |
| `src/quant_research/validation/report_renderer.py` | `render_structured_report_markdown`, `render_structured_report_html`, `render_structured_report`, `write_structured_report_artifact` | Renders and writes structured report sections that may contain action aggregates. | Generic renderer; it does not compute labels. |
| `src/quant_research/validation/gate.py` | `ValidationGateReport.to_markdown` | Renders no-model-proxy `Buy / Sell / Hold` aggregate counts when available. | Validity report aggregate display only. |
| `src/quant_research/validation/gate.py` | `ValidationGateReport.to_html` | Renders no-model-proxy `Buy / Sell / Hold` aggregate counts when available. | HTML counterpart to the Markdown validity report surface. |

## Export Surfaces

| Path | Function or object | Emission | Notes |
|---|---|---|---|
| `scripts/run_backtest_validation.py` | `save_outputs` | Writes `result.backtest.signals` to `signals.csv`. | CSV export of deterministic backtest signals, including `action`. |
| `scripts/run_backtest_validation.py` | `save_outputs` | Builds completed reports from `result.backtest.signals`. | Report artifacts derive action aggregates from deterministic signals. |
| `scripts/run_backtest_validation.py` | `save_outputs` | Includes `signals.csv` as `deterministic_signals` in artifact manifests. | Manifest reference only; no label generation. |
| `app.py` | validity report download buttons | Downloads `validity_gate.json` and `validity_report.md`. | Downloads may include deterministic aggregate action counts from the validity report. |

## Explicit Non-Emission Paths

| Path | Function or object | Why it is not a final label emission path |
|---|---|---|
| `src/quant_research/models/` | model adapters | Produce structured numeric/text features only; they do not create final `action` labels. |
| `src/quant_research/features/` | feature builders | Build price, text, SEC, and fused features only. |
| `src/quant_research/validation/gate.py` | validity gate metrics and report-only research metrics | Gate and report-only metrics may summarize existing labels but do not change scores or assign `BUY`, `SELL`, or `HOLD`. |
| `tests/fixtures/report_generation/signals.csv` | report fixture | Test fixture for report generation, not a production emission path. |

## Audit Commands

Run these commands from the repository root to reproduce the inventory search:

```bash
rg -n "\\b(BUY|SELL|HOLD|action_counts|latest_action_counts|\\[\"action\"\\]|\\.action)\\b" app.py scripts src tests docs
rg -n "signals\\.to_csv|result\\.signals|backtest\\.signals|deterministic_signal_outputs|raw_signal|Latest Deterministic Signals" app.py scripts src tests docs
python3 -m pytest tests/test_architecture_guards.py
```

Expected invariant: only `src/quant_research/signals/engine.py` assigns final
row-level `action` labels; UI, report, export, and package API paths only
display, aggregate, serialize, or pass through those labels.
