# Technical report — silicone compression-set data for Los Alamos collaborators

Overleaf-ready LaTeX report describing the extraction pipeline, the frozen
release `cs_v14` (1,054 measurements / 605 formulations / 67 documents), the
predictive baseline on it, and the companion Noll table benchmark.

## Layout

| Path | What |
|---|---|
| `main.tex` | The progress-report-style report (executive summary, goals, accomplishments, significant results, next steps, and appendices). `article`, natbib + bibtex, Helvetica text. No `biblatex`, `siunitx`, `authblk`, or `tcolorbox`, so it compiles on the CRC TeX Live as well as on Overleaf. |
| `references.bib` | Bibliography (shared with `rtv_foam_spr/research_paper/`). |
| `make_figures.py` | Regenerates the five figures **and** `generated/numbers.tex` from committed artifacts. |
| `generated/numbers.tex` | One `\newcommand` per number the prose quotes (`\nRelRows`, `\nLofoConditions`, …). Generated; never edit. |
| `figures/` | Each figure as `.pdf` + `.png` + `.svg` on one descriptive stem. `main.tex` includes the `.png`. |
| `scripts/qsub/make_figures.sh` | The job that runs `make_figures.py` (rtv venv, CPU). |

## Figures

1. `problem_qualitative_record_vs_quantitative_compression_set_data` — prominent corpus funnel (documents → clean release), row-stated vs document-resolved condition coverage, and composite/foam formulation classes.
2. `pipeline_from_source_discovery_to_grouped_evaluation` — three-lane matplotlib diagram (acquisition / extraction and resolution / release and evaluation).
3. `release_cs_v14_target_sources_amount_basis_roles_protocols_and_field_coverage` — six-panel statistics of the frozen release.
4. `ladder_conditions_floor_vs_chemistry_arms_grouped_random_and_paired_gain` — absolute R² under three regimes plus paired gain with interval and smallest provable gain.
5. `umap_of_formulation_embeddings_by_document_cure_system_and_compression_set` — UMAP of the 768-d amount-weighted ingredient-text embeddings.

## Inputs (all committed)

- `3_machine_learning_model/data/cs_v14/` (snapshot: `release/*.csv`, `measurements.csv`, `components.csv`, `formulation_blocks.csv`, `summary.json`, `manifest.json`)
- `3_machine_learning_model/results/cs_v14/summary.csv` (the ladder)
- `rtv_foam_spr/data/rtv_foam.duckdb` (corpus funnel and condition coverage; same build as the release)

The report's operational formulation labels are derived from `formulations.csv`:
composite means at least one reinforcing, functional, extending, or lightweight
filler role; foam means a blowing-agent role. The labels overlap, so filled
foams are counted in both classes.

## Build

```bash
cd technical_report
qsub scripts/qsub/make_figures.sh      # figures + generated/numbers.tex (a few minutes)
make                                   # main.pdf via pdflatex → bibtex → pdflatex ×2
make overleaf                          # zip with exactly the files Overleaf needs
```

Upload `los_alamos_technical_report_overleaf.zip` through Overleaf's *New Project → Upload Project*; the main file is `main.tex`, compiler pdfLaTeX. Or push this folder to a git remote and use Overleaf's GitHub sync.

## Conventions the text follows

- Numbers in prose come from macros in `generated/numbers.tex`; the only literals typed by hand are the audit percentages (from `rtv_foam_spr/data/gold/table_fidelity_audit/cs_v14/summary.md`), the release history, the acquisition-pipeline counts (early September 2026), and the Noll benchmark counts.
- Evaluation vocabulary: "grouped" and "in-distribution"; gain uncertainty is "from leaving out one document family at a time"; the detectable effect is the "smallest provable gain".
- cs_v14 = 1,054 / 67 / 605. Do not quote 0.251, 903/67, or 933/72 anywhere.
