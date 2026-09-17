"""Figures and number macros for the Los Alamos technical report.

    python make_figures.py            # all figures + generated/numbers.tex
    python make_figures.py problem    # one figure by stem

Every figure is read from committed project artifacts (the cs_v14 snapshot
under 3_machine_learning_model/data/cs_v14/, the ladder metrics under
3_machine_learning_model/results/cs_v14/, and the production DuckDB for the
corpus-wide funnel; that funnel begins with the downloaded-PDF inventory) and
written as PDF + PNG + SVG on one stem under
figures/. Numbers the report quotes are emitted to generated/numbers.tex as
LaTeX macros so the prose can never drift from the data.

House style (.claude/skills/academic-figure-plotting): Helvetica, every text
element bold and >= 16 pt, no title drawn on the figure, ball-and-stick marks
in place of filled bars, three export formats on the same stem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RTV = ROOT / "rtv_foam_spr"
ML = ROOT / "3_machine_learning_model"
SNAP = ML / "data" / "cs_v14"
RELEASE = SNAP / "release"
RESULTS = ML / "results" / "cs_v14"
FIG = HERE / "figures"
GEN = HERE / "generated"
FIG.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / ".claude" / "skills" / "academic-figure-plotting" / "scripts"))
from plot_utils import MIN_FONTSIZE, save_all_formats, setup_style, style_axes  # noqa: E402

# Register the group's Helvetica files and drop the narrow/light variants that
# matplotlib's matcher would otherwise pick up under the same family name.
_FONT_DIR = Path("/users/bpiguave/.local/share/fonts")
for _name in ("Helvetica.ttf", "Helvetica-Bold.ttf"):
    if (_FONT_DIR / _name).exists():
        font_manager.fontManager.addfont(str(_FONT_DIR / _name))
font_manager.fontManager.ttflist = [
    f for f in font_manager.fontManager.ttflist
    if not any(s in Path(f.fname).name.lower()
               for s in ("condensed", "compressed", "rounded", "light", "oblique"))
]
setup_style(fontsize=18)

NAVY, BLUE, ORANGE, TEAL, PURPLE, GREY, GOLD = (
    "#0b2e4f", "#1565c0", "#ef6c00", "#00838f", "#6a1b9a", "#9aa8b5", "#c99700")
FS_LABEL, FS_TICK, FS_ANNOT, FS_PANEL = 22, 18, 17, 24
NUMBERS: dict[str, str] = {}

# The report starts at the downloaded-PDF inventory.  The papers table also
# contains a larger candidate ledger, but those candidates were not processed
# and are outside the scope of the report.
DOWNLOADED_PDFS = 2925


def num(key: str, value, fmt: str = "{:,}") -> None:
    """Record a number the report quotes; written to generated/numbers.tex."""
    if not key.isalpha():
        raise ValueError(f"macro key must be letters only (LaTeX): {key!r}")
    NUMBERS[key] = fmt.format(value) if not isinstance(value, str) else value


def panel(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left")


def clean(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#d9dee3", lw=0.8)


def lollipop(ax, pos, vals, color, horizontal=False, stick_lw=9, ms=13, label=None, alpha=0.45):
    pos, vals = np.asarray(pos, float), np.asarray(vals, float)
    if horizontal:
        ax.hlines(pos, 0, vals, color=color, lw=stick_lw, alpha=alpha, zorder=2)
        ax.plot(vals, pos, "o", color=color, ms=ms, zorder=3, label=label)
    else:
        ax.vlines(pos, 0, vals, color=color, lw=stick_lw, alpha=alpha, zorder=2)
        ax.plot(pos, vals, "o", color=color, ms=ms, zorder=3, label=label)


def bold_ticks(ax) -> None:
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_fontsize(FS_TICK)


def save(fig, stem: str) -> None:
    paths = save_all_formats(fig, str(FIG), stem)
    plt.close(fig)
    print(f"[{stem}] " + ", ".join(Path(p).name for p in paths.values()))


# --------------------------------------------------------------------------
# shared release loaders
# --------------------------------------------------------------------------
def load_release():
    cs = pd.read_csv(RELEASE / "compression_set.csv")
    forms = pd.read_csv(RELEASE / "formulations.csv")
    docs = pd.read_csv(RELEASE / "documents.csv")
    meas = pd.read_csv(SNAP / "measurements.csv")
    summary = json.loads((SNAP / "summary.json").read_text())
    return cs, forms, docs, meas, summary


# --------------------------------------------------------------------------
# Figure 1 -- the problem: a large qualitative record, a small quantitative one
# --------------------------------------------------------------------------
def fig_problem():
    import duckdb

    sys.path.insert(0, str(RTV))
    from pipeline.ml.data.conditions import parse_conditions

    con = duckdb.connect(str(RTV / "data" / "rtv_foam.duckdb"), read_only=True)
    q = lambda s: con.execute(s).fetchone()  # noqa: E731
    n_docs = DOWNLOADED_PDFS
    n_prop_docs = q("select count(distinct ref_id) from properties")[0]
    n_cs_docs, n_cs_rows = q(
        "select count(distinct ref_id), count(*) from properties where property_canonical='compression_set'")
    base = ("from properties p where property_canonical='compression_set' and not coalesce(is_prophetic,false)"
            " and coalesce(in_range,true)")
    comp = " and exists(select 1 from ingredients i where i.formulation_id=p.formulation_id)"
    cond = " and cond_aging_temp_c is not null and cond_deflection_pct is not null and cond_aging_time_h is not null"
    n_comp_docs, n_comp_rows = q(f"select count(distinct p.ref_id), count(*) {base}{comp}")
    n_cond_docs, n_cond_rows = q(f"select count(distinct p.ref_id), count(*) {base}{comp}{cond}")
    cs_release, forms_release, _, _, summary = load_release()
    n_rel_docs, n_rel_rows = summary["documents"], summary["measurements"]

    # Panel (b): the same compression-set rows read two ways -- what the
    # measurement row itself states, and what the database carries after
    # document-level resolution of the methods text.
    cs_db = con.execute(f"""
        select property_verbatim, measurement_conditions,
               (value_canonical is not null or value_verbatim is not null) as has_value,
               exists(select 1 from ingredients i where i.formulation_id=p.formulation_id) as has_ing,
               cond_deflection_pct is not null as db_defl,
               cond_aging_temp_c is not null as db_temp,
               cond_aging_time_h is not null as db_time
        {base}""").fetchall()
    con.close()
    n = len(cs_db)
    row = dict(value=0, ingredients=0, temp=0, time=0, deflection=0, all=0)
    res = dict(row)
    for pv, mc, hv, hi, ddb, tdb, tim in cs_db:
        c = parse_conditions(pv or "", mc or "")
        d = c["deflection_pct"] is not None
        t = (c["test_temp_C"] is not None) or (c["aging_temp_C"] is not None)
        h = (c.get("test_time_h") is not None) or (c.get("aging_time_h") is not None)
        for k, v in (("value", hv), ("ingredients", hi), ("temp", t), ("time", h), ("deflection", d),
                     ("all", hv and hi and t and h and d)):
            row[k] += bool(v)
        for k, v in (("value", hv), ("ingredients", hi), ("temp", tdb), ("time", tim), ("deflection", ddb),
                     ("all", hv and hi and tdb and tim and ddb)):
            res[k] += bool(v)
    row_pct = {k: 100 * v / n for k, v in row.items()}
    res_pct = {k: 100 * v / n for k, v in res.items()}

    # Panel (c): formulation classes derived from the reviewed role taxonomy.
    forms = forms_release
    filler_roles = {"reinforcing_filler", "functional_filler", "extending_filler",
                    "lightweight_filler", "other_filler"}
    roles_by_form = forms.groupby("formulation_id")["role"].agg(lambda s: set(s))
    is_composite = roles_by_form.map(lambda s: bool(s & filler_roles))
    is_foam = roles_by_form.map(lambda s: "blowing_agent" in s)
    class_order = ["composite only", "composite + foam", "foam only", "neither"]
    class_by_form = pd.Series("neither", index=roles_by_form.index)
    class_by_form[is_composite & ~is_foam] = "composite only"
    class_by_form[is_composite & is_foam] = "composite + foam"
    class_by_form[~is_composite & is_foam] = "foam only"
    class_counts = class_by_form.value_counts().reindex(class_order).fillna(0).astype(int)
    cs_class = cs_release["formulation_id"].map(class_by_form)
    class_rows = cs_class.value_counts().reindex(class_order).fillna(0).astype(int)
    composite_docs = forms.loc[forms.formulation_id.isin(roles_by_form[is_composite].index), "doc_id"].nunique()
    foam_docs = forms.loc[forms.formulation_id.isin(roles_by_form[is_foam].index), "doc_id"].nunique()
    comp_src = pd.read_csv(SNAP / "components.csv", usecols=["formulation_id", "name_verbatim", "category_labels"])
    comp_src["role"] = comp_src["category_labels"].fillna("").str.split("|").str[0]
    spell = comp_src.groupby("role").name_verbatim.nunique().sort_values(ascending=False)
    spell = spell[[r for r in spell.index if r and r != "unknown_role"]].head(7)

    fig = plt.figure(figsize=(22, 10.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # (a) funnel: documents at each stage of the compression-set record
    ax = axes[0]
    stages = [
        ("downloaded PDFs", n_docs, None),
        ("any extracted property", n_prop_docs, None),
        ("any compression-set value", n_cs_docs, n_cs_rows),
        ("value with a composition", n_comp_docs, n_comp_rows),
        ("value, composition,\nall three test conditions", n_cond_docs, n_cond_rows),
        ("clean release cs_v14", n_rel_docs, n_rel_rows),
    ]
    y = np.arange(len(stages))[::-1]
    vals = [s[1] for s in stages]
    colors = [GREY, GREY, BLUE, BLUE, BLUE, ORANGE]
    for yi, v, c in zip(y, vals, colors):
        lollipop(ax, [yi], [v], c, horizontal=True, stick_lw=11, ms=15)
    ax.set_xscale("log")
    ax.set_xlim(10, 12000)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in stages])
    for yi, (label, v, rows) in zip(y, stages):
        txt = f"{v:,}" + (f" docs / {rows:,} rows" if rows else " docs")
        ax.text(v * 1.25, yi, txt, va="center", ha="left", fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    style_axes(ax, xlabel="documents (log scale)", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    clean(ax, "x")
    bold_ticks(ax)
    panel(ax, "a", x=-0.04, y=1.03)

    # (b) what the measurement row states vs what document-level reading recovers
    ax = axes[1]
    keys = ["value", "ingredients", "temp", "time", "deflection", "all"]
    names = ["value", "composition", "temperature", "time", "deflection", "all fields"]
    y = np.arange(len(keys))[::-1]
    lollipop(ax, y + 0.2, [row_pct[k] for k in keys], GREY, horizontal=True, stick_lw=9, ms=12,
             label="stated in the measurement row")
    lollipop(ax, y - 0.2, [res_pct[k] for k in keys], BLUE, horizontal=True, stick_lw=9, ms=12,
             label="after document-level resolution")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 118)
    for k, yi in zip(keys, y):
        ax.text(row_pct[k] + 2.5, yi + 0.2, f"{row_pct[k]:.0f}%", va="center", fontsize=MIN_FONTSIZE,
                fontweight="bold", color="#5c6770")
        ax.text(res_pct[k] + 2.5, yi - 0.2, f"{res_pct[k]:.0f}%", va="center", fontsize=MIN_FONTSIZE,
                fontweight="bold", color=BLUE)
    style_axes(ax, xlabel=f"% of {n:,} compression-set rows", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=1,
              prop=dict(weight="bold", size=MIN_FONTSIZE), handletextpad=0.4)
    clean(ax, "x")
    bold_ticks(ax)
    panel(ax, "b", x=-0.14, y=1.04)

    # (c) formulation classes: composites and foams
    ax = axes[2]
    y = np.arange(len(class_order))[::-1]
    class_colors = [BLUE, PURPLE, ORANGE, GREY]
    for yi, name, v, c in zip(y, class_order, class_counts, class_colors):
        lollipop(ax, [yi], [v], c, horizontal=True, stick_lw=13, ms=16)
        ax.text(v + class_counts.max() * 0.025, yi,
                f"{v:,} forms / {class_rows[name]:,} measurements",
                va="center", fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(class_order)
    ax.set_xlim(0, class_counts.max() * 1.42)
    style_axes(ax, xlabel="formulations (n = 605)", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    clean(ax, "x")
    bold_ticks(ax)
    panel(ax, "c", x=-0.14, y=1.04)

    save(fig, "problem_qualitative_record_vs_quantitative_compression_set_data")

    num("CorpusDocs", n_docs); num("PropDocs", n_prop_docs)
    num("CsDocs", n_cs_docs); num("CsRows", n_cs_rows)
    num("CsCompDocs", n_comp_docs); num("CsCompRows", n_comp_rows)
    num("CsCondDocs", n_cond_docs); num("CsCondRows", n_cond_rows)
    num("CsPopulation", n)
    num("CompositeForms", int(is_composite.sum()))
    num("FoamForms", int(is_foam.sum()))
    num("CompositeDocs", int(composite_docs))
    num("FoamDocs", int(foam_docs))
    num("CompositeRows", int(cs_release["formulation_id"].map(is_composite).fillna(False).sum()))
    num("FoamRows", int(cs_release["formulation_id"].map(is_foam).fillna(False).sum()))
    class_macro_keys = {
        "composite only": ("CompositeOnlyForms", "CompositeOnlyRows"),
        "composite + foam": ("CompositeFoamForms", "CompositeFoamRows"),
        "foam only": ("FoamOnlyForms", "FoamOnlyRows"),
        "neither": ("NeitherForms", "NeitherRows"),
    }
    for label, (form_key, row_key) in class_macro_keys.items():
        num(form_key, int(class_counts[label]))
        num(row_key, int(class_rows[label]))
    for k in keys:
        num(f"Row{k.capitalize()}Pct", round(row_pct[k]), "{}")
        num(f"Res{k.capitalize()}Pct", round(res_pct[k]), "{}")
    num("SpellTopRole", spell.index[0].replace("_", " "), "{}")
    num("SpellTopCount", int(spell.iloc[0]))
    num("ReleaseComponentRows", len(forms))
    num("ReleaseDistinctStrings", forms["ingredient"].nunique())


# --------------------------------------------------------------------------
# Figure 2 -- the pipeline, drawn with matplotlib patches
# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec=NAVY, tc="white", fs=MIN_FONTSIZE, lw=2.0, style="round,pad=0.02,rounding_size=0.12"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc,
            zorder=3, linespacing=1.25)


def _arrow(ax, p0, p1, color=NAVY, lw=2.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=22, color=color, lw=lw, ls=ls, zorder=4,
                                 shrinkA=2, shrinkB=2))


def fig_pipeline():
    _, _, _, _, summary = load_release()
    n_rel_docs, n_rel_rows, n_rel_forms = summary["documents"], summary["measurements"], summary["formulations"]
    fig, ax = plt.subplots(figsize=(22, 10.5), layout="constrained")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    lane_fc = ["#eef4fb", "#f3eefa", "#eaf6f5"]
    lanes = ["1  Acquisition", "2  Extraction and resolution", "3  Release and evaluation"]
    for i, (lab, fc) in enumerate(zip(lanes, lane_fc)):
        y0 = 7.2 - 3.5 * i
        ax.add_patch(FancyBboxPatch((0.25, y0), 21.5, 3.15, boxstyle="round,pad=0.02,rounding_size=0.2", fc=fc,
                                    ec="none", zorder=1))
        ax.text(0.5, y0 + 2.85, lab, fontsize=FS_LABEL, fontweight="bold", color=NAVY, va="center")

    # lane 1 -- acquisition
    y = 7.55
    _box(ax, 0.6, y, 3.6, 1.6, "Query specification\nmaterial + form terms,\nrendered per source", BLUE)
    _box(ax, 4.9, y, 3.8, 1.6, "Sources\nUSPTO, EPO OPS, OpenAlex,\nSemantic Scholar, OSTI", BLUE)
    _box(ax, 9.4, y, 3.6, 1.6, "Permanent doc_id\n+ identifier ledger\n(DOI, family, hash)", NAVY)
    _box(ax, 13.7, y, 3.6, 1.6, "Document parsing\npage image to text\n(vision-language)", PURPLE)
    _box(ax, 18.0, y, 3.5, 1.6, "Two-stage screen\nsiloxane matrix?\ncompression set?", TEAL)
    for x0, x1 in ((4.2, 4.9), (8.7, 9.4), (13.0, 13.7), (17.3, 18.0)):
        _arrow(ax, (x0, y + 0.8), (x1, y + 0.8))

    # lane 2 -- extraction & resolution
    y = 4.05
    _box(ax, 0.6, y, 3.6, 1.6, "Page-image extraction\nschema-enforced record,\nvalues as printed", PURPLE)
    _box(ax, 4.9, y, 3.8, 1.6, "Ingredient resolution\ncanonical substance,\nrole taxonomy (LLM)", PURPLE)
    _box(ax, 9.4, y, 3.6, 1.6, "Condition resolution\ntemperature, time,\ndeflection from methods", PURPLE)
    _box(ax, 13.7, y, 3.6, 1.6, "Evidence check\nevery value against a\nverbatim source quote", ORANGE)
    _box(ax, 18.0, y, 3.5, 1.6, "Curated corrections\nhuman > agent >\nextraction", ORANGE)
    for x0, x1 in ((4.2, 4.9), (8.7, 9.4), (13.0, 13.7), (17.3, 18.0)):
        _arrow(ax, (x0, y + 0.8), (x1, y + 0.8))

    # lane 3 -- release & evaluation
    y = 0.55
    _box(ax, 0.6, y, 3.6, 1.6, "Rebuilt database\nDuckDB, one schema,\nreproducible from inputs", NAVY)
    _box(ax, 4.9, y, 3.8, 1.6, f"Frozen release cs_v14\n{n_rel_rows:,} measurements\n{n_rel_forms} formulations, {n_rel_docs} docs", GOLD, tc=NAVY)
    _box(ax, 9.4, y, 3.6, 1.6, "Feature arms\nconditions, descriptors,\nembeddings, categories", BLUE)
    _box(ax, 13.7, y, 3.6, 1.6, "Grouped evaluation\npatent family held out,\nleave one family out", BLUE)
    _box(ax, 18.0, y, 3.5, 1.6, "Report\nabsolute R2, paired gain,\nsmallest provable gain", TEAL)
    for x0, x1 in ((4.2, 4.9), (8.7, 9.4), (13.0, 13.7), (17.3, 18.0)):
        _arrow(ax, (x0, y + 0.8), (x1, y + 0.8))

    # lane transitions (right edge down, left edge down)
    _arrow(ax, (19.75, 7.55), (19.75, 6.35), color=NAVY, lw=2.8)
    ax.text(20.0, 6.95, "screened PDFs", fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY, va="center")
    _arrow(ax, (19.75, 4.05), (19.75, 2.85), color=NAVY, lw=2.8)
    ax.text(20.0, 3.45, "candidate rows", fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY, va="center")
    # feedback: evaluation gaps back to extraction
    _arrow(ax, (15.5, 2.15), (15.5, 4.05), color=ORANGE, lw=2.2, ls="--")
    ax.text(15.7, 3.1, "audit findings\nfix the stage,\nrerun", fontsize=MIN_FONTSIZE, fontweight="bold",
            color=ORANGE, va="center")
    save(fig, "pipeline_from_source_discovery_to_grouped_evaluation")


# --------------------------------------------------------------------------
# Figure 3 -- what the frozen release contains
# --------------------------------------------------------------------------
def fig_dataset():
    cs, forms, docs, meas, summary = load_release()
    fig, axes = plt.subplots(2, 3, figsize=(22, 13.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.3, h_pad=0.4, wspace=0.08, hspace=0.08)

    # (a) target distribution
    ax = axes[0, 0]
    ax.hist(cs["compression_set_pct"], bins=np.arange(0, 105, 5), color=BLUE, edgecolor="white", lw=1.2)
    med = cs["compression_set_pct"].median()
    ax.axvline(med, color=ORANGE, lw=3, ls="--")
    ax.text(med + 2, ax.get_ylim()[1] * 0.92, f"median {med:.0f}%", color=ORANGE, fontsize=FS_ANNOT, fontweight="bold")
    style_axes(ax, xlabel="compression set (%)", ylabel="measurements", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    clean(ax); bold_ticks(ax); panel(ax, "a")

    # (b) measurements per document, sorted, journal articles marked
    ax = axes[0, 1]
    d = docs.sort_values("n_measurements", ascending=False).reset_index(drop=True)
    cols = [ORANGE if s == "journal" else TEAL for s in d["source_type"]]
    ax.bar(np.arange(len(d)), d["n_measurements"], color=cols, width=0.85)
    ax.plot([], [], color=TEAL, lw=10, label=f"patent ({(d.source_type == 'patent').sum()})")
    ax.plot([], [], color=ORANGE, lw=10, label=f"journal ({(d.source_type == 'journal').sum()})")
    ax.legend(frameon=False, prop=dict(weight="bold", size=MIN_FONTSIZE))
    ax.text(len(d) * 0.55, d["n_measurements"].max() * 0.55,
            f"median {int(d.n_measurements.median())} per document\nlargest {int(d.n_measurements.max())}",
            fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    ax.set_xticks([])
    style_axes(ax, xlabel=f"document (n = {len(d)}, sorted)", ylabel="measurements", label_fontsize=FS_LABEL,
               tick_fontsize=FS_TICK)
    clean(ax); bold_ticks(ax); panel(ax, "b")

    # (c) amount basis per formulation
    ax = axes[0, 2]
    pretty = {"parts_by_weight": "parts per hundred", "wt_percent_final_formulation": "weight %",
              "wt_pct": "weight %", "mass_g": "grams", "unspecified": "unspecified", "ppm": "ppm",
              "wt_fraction": "weight fraction", "parts_by_mass": "parts per hundred"}
    # One basis per formulation. ppm-level entries (trace catalyst) and rows
    # without a basis do not make a recipe "mixed"; a recipe is mixed only when
    # two weighable bases (e.g. parts and weight %) appear together.
    def one_basis(s):
        v = sorted({pretty.get(x, x) for x in s.dropna() if x not in ("ppm", "unspecified")})
        if not v:
            return "unspecified"
        return "mixed" if len(v) > 1 else v[0]
    per_form = forms.groupby("formulation_id")["amount_basis"].agg(one_basis)
    basis = per_form.value_counts()
    order = [b for b in ["parts per hundred", "weight %", "grams", "mixed", "unspecified"] if b in basis.index]
    basis = basis[order]
    y = np.arange(len(basis))[::-1]
    lollipop(ax, y, basis.values, BLUE, horizontal=True, stick_lw=11, ms=15)
    ax.set_yticks(y); ax.set_yticklabels(basis.index)
    for yi, v in zip(y, basis.values):
        ax.text(v + basis.max() * 0.05, yi, f"{v}", va="center", fontsize=FS_ANNOT, fontweight="bold", color=NAVY)
    ax.set_xlim(0, basis.max() * 1.25)
    style_axes(ax, xlabel=f"formulations (n = {per_form.size})", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    clean(ax, "x"); bold_ticks(ax); panel(ax, "c", x=-0.42)
    for k, v in basis.items():
        num("Basis" + "".join(w.capitalize() for w in k.replace("%", "pct").split()), int(v))

    # (d) role prevalence: share of formulations naming each role
    ax = axes[1, 0]
    roles = ["base_polymer", "crosslinker", "cure_catalyst", "cure_initiator", "cure_inhibitor",
             "reinforcing_filler", "extending_filler", "functional_filler", "filler_treating_agent",
             "plasticizer_or_extender_oil", "blowing_agent", "pigment"]
    nf = forms["formulation_id"].nunique()
    prev = forms.groupby("role")["formulation_id"].nunique().reindex(roles).fillna(0) / nf * 100
    y = np.arange(len(roles))[::-1]
    lollipop(ax, y, prev.values, TEAL, horizontal=True, stick_lw=9, ms=13)
    ax.set_yticks(y); ax.set_yticklabels([r.replace("_or_", " / ").replace("_", " ") for r in roles])
    ax.set_xlim(0, 108)
    for yi, v in zip(y, prev.values):
        ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY)
    style_axes(ax, xlabel="% of formulations naming the role", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    clean(ax, "x"); bold_ticks(ax); panel(ax, "d", x=-0.62)
    num("RoleBasePolymerPct", round(prev["base_polymer"]), "{}")
    num("RoleCrosslinkerPct", round(prev["crosslinker"]), "{}")
    num("RoleCatalystPct", round(prev["cure_catalyst"]), "{}")
    num("RoleInitiatorPct", round(prev["cure_initiator"]), "{}")
    num("RoleReinforcingFillerPct", round(prev["reinforcing_filler"]), "{}")

    # (e) test protocol: temperature vs time, deflection as marker size
    ax = axes[1, 1]
    t = cs["time_h"].clip(lower=0.1)
    sizes = 40 + 6 * cs["deflection_pct"].fillna(25)
    ax.scatter(t, cs["temperature_C"], s=sizes, c=BLUE, alpha=0.25, edgecolor="none")
    ax.set_xscale("log")
    for lab, tt, TT, off in (("22 h / 175 °C", 22, 175, (-80, 75)), ("70 h / 150 °C", 70, 150, (70, -80)),
                             ("24 h / 23 °C", 24, 23, (45, 25))):
        n_here = int(((cs.time_h == tt) & (cs.temperature_C == TT)).sum())
        if n_here:
            ax.annotate(f"{lab}\nn = {n_here}", (tt, TT), xytext=off, textcoords="offset points",
                        fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY,
                        arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.2))
    style_axes(ax, xlabel="test time (h, log scale)", ylabel="test temperature (°C)", label_fontsize=FS_LABEL,
               tick_fontsize=FS_TICK)
    clean(ax, "both"); bold_ticks(ax); panel(ax, "e")
    num("ProtocolDistinct", int(cs.groupby(["temperature_C", "time_h", "deflection_pct"]).ngroups))
    num("ProtocolTwentyTwoHOneSevenFive", int(((cs.time_h == 22) & (cs.temperature_C == 175)).sum()))
    num("ProtocolSeventyHOneFifty", int(((cs.time_h == 70) & (cs.temperature_C == 150)).sum()))

    # (f) which measurement fields the source actually states
    ax = axes[1, 2]
    n = len(cs)
    fields = [("temperature", cs.temperature_C), ("time", cs.time_h), ("deflection", cs.deflection_pct),
              ("medium", cs.medium), ("specimen shape", cs.specimen_shape),
              ("specimen thickness", cs.specimen_thickness_mm), ("specimen diameter", cs.specimen_diameter_mm),
              ("test standard", cs.test_standard)]
    stated = [100 * s.notna().mean() for _, s in fields]
    y = np.arange(len(fields))[::-1]
    lollipop(ax, y, stated, BLUE, horizontal=True, stick_lw=9, ms=13)
    ax.set_yticks(y); ax.set_yticklabels([f for f, _ in fields])
    ax.set_xlim(0, 118)
    for yi, v in zip(y, stated):
        ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY)
    style_axes(ax, xlabel="field stated (% of rows)", label_fontsize=FS_LABEL,
               tick_fontsize=FS_TICK)
    clean(ax, "x"); bold_ticks(ax); panel(ax, "f", x=-0.5)
    for (f, _), v in zip(fields, stated):
        num("Stated" + "".join(w.capitalize() for w in f.split()), round(v), "{}")

    save(fig, "release_cs_v14_target_sources_amount_basis_roles_protocols_and_field_coverage")

    num("RelDocs", summary["documents"]); num("RelRows", summary["measurements"])
    num("RelForms", summary["formulations"]); num("RelGroups", summary["cv_groups"])
    num("RelPatents", summary["source_type_docs"]["patent"]); num("RelJournals", summary["source_type_docs"]["journal"])
    num("RelMedian", med, "{:.0f}"); num("RelMean", cs.compression_set_pct.mean(), "{:.1f}")
    num("RelIQRlo", cs.compression_set_pct.quantile(0.25), "{:.0f}")
    num("RelIQRhi", cs.compression_set_pct.quantile(0.75), "{:.0f}")
    num("DocMedianRows", int(docs.n_measurements.median())); num("DocMaxRows", int(docs.n_measurements.max()))
    num("RelIngredientRows", len(forms)); num("RelDistinctIngredients", forms.ingredient.nunique())
    num("RelYearMin", int(docs.year.min()), "{}"); num("RelYearMax", int(docs.year.max()), "{}")


# --------------------------------------------------------------------------
# Figure 4 -- the ladder: does chemistry add to the conditions floor?
# --------------------------------------------------------------------------
ARM_KEY = {"conditions": "Conditions", "descriptors_visc": "DescriptorsVisc", "emb_weighted": "EmbWeighted",
           "blend_embdesc_50": "BlendEmbdescFifty", "categories": "Categories"}
LADDER = [("ridge", "conditions", "conditions\nonly"),
          ("xgboost", "descriptors_visc", "+ stated\nviscosity"),
          ("xgboost", "emb_weighted", "+ ingredient\ntext"),
          ("xgboost", "blend_embdesc_50", "+ text and\ndescriptors"),
          ("xgboost", "categories", "+ ingredient\ncategories")]


def fig_ladder():
    m = pd.read_csv(RESULTS / "summary.csv")
    rows = [m[(m.estimator == e) & (m.arm == a)].iloc[0] for e, a, _ in LADDER]
    labels = [lab for _, _, lab in LADDER]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.2), layout="constrained", gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    fig.get_layout_engine().set(w_pad=0.3, wspace=0.1)

    # (a) absolute R2 under three evaluation regimes
    ax = axes[0]
    off = 0.26
    lollipop(ax, x - off, [r.random_R2 for r in rows], GREY, stick_lw=10, ms=14, label="random rows (in-distribution)")
    lollipop(ax, x, [r.grouped_R2 for r in rows], BLUE, stick_lw=10, ms=14, label="grouped five-fold (patent family)")
    ax.errorbar(x, [r.grouped_R2 for r in rows], yerr=[r.grouped_R2_sd for r in rows], fmt="none", ecolor=BLUE,
                elinewidth=2, capsize=5, zorder=4)
    lollipop(ax, x + off, [r.lofo_R2 for r in rows], ORANGE, stick_lw=10, ms=14, label="leave one family out")
    floor = rows[0].lofo_R2
    ax.axhline(floor, color=ORANGE, lw=1.8, ls="--", zorder=1, label=f"conditions floor (LOFO {floor:.2f})")
    for xi, r in zip(x, rows):
        ax.text(xi - off, r.random_R2 + 0.03, f"{r.random_R2:.2f}", ha="center", fontsize=MIN_FONTSIZE,
                fontweight="bold", color="#5c6770")
        ax.text(xi + off, r.lofo_R2 + 0.03, f"{r.lofo_R2:.2f}", ha="center", fontsize=MIN_FONTSIZE,
                fontweight="bold", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, len(rows) - 0.4)
    ax.set_ylim(0, 0.85)
    style_axes(ax, ylabel="R² on held-out rows", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    ax.legend(loc="upper left", frameon=False, prop=dict(weight="bold", size=MIN_FONTSIZE), ncol=1)
    clean(ax); bold_ticks(ax); panel(ax, "a", x=-0.08)

    # (b) paired gain over the conditions floor with its interval and the
    #     smallest gain this corpus could prove
    ax = axes[1]
    chem = rows[1:]
    y = np.arange(len(chem))[::-1]
    for yi, r in zip(y, chem):
        ax.plot([r.ci95_low, r.ci95_high], [yi, yi], color=BLUE, lw=4, solid_capstyle="round", zorder=2)
        ax.plot(r.delta_lofo_R2, yi, "o", color=BLUE, ms=15, zorder=3)
        ax.plot(r.smallest_provable_gain, yi, "D", mfc="white", mec=ORANGE, mew=2.6, ms=13, zorder=3)
        ax.text(r.delta_lofo_R2, yi + 0.2, f"{r.delta_lofo_R2:+.2f}", va="bottom", ha="center",
                fontsize=MIN_FONTSIZE, fontweight="bold", color=BLUE)
    ax.axvline(0, color=NAVY, lw=2)
    ax.plot([], [], "o", color=BLUE, ms=13, label="gain in R² over conditions [95% interval]")
    ax.plot([], [], "D", mfc="white", mec=ORANGE, mew=2.6, ms=12, label="smallest provable gain at 67 documents")
    ax.set_yticks(y); ax.set_yticklabels([lab.replace("\n", " ") for _, _, lab in LADDER[1:]])
    ax.set_xlim(-0.18, 0.36)
    ax.set_ylim(-0.7, len(chem) - 0.3)
    style_axes(ax, xlabel="gain in leave-one-family-out R²\nover the conditions floor",
               label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), frameon=False, ncol=1,
              prop=dict(weight="bold", size=MIN_FONTSIZE))
    clean(ax, "x"); bold_ticks(ax); panel(ax, "b", x=-0.62)

    save(fig, "ladder_conditions_floor_vs_chemistry_arms_grouped_random_and_paired_gain")

    for (e, a, _), r in zip(LADDER, rows):
        key = ARM_KEY[a]
        num(f"Grouped{key}", r.grouped_R2, "{:.3f}"); num(f"Random{key}", r.random_R2, "{:.3f}")
        num(f"Lofo{key}", r.lofo_R2, "{:.3f}"); num(f"LofoMae{key}", r.lofo_MAE, "{:.1f}")
        if pd.notna(r.delta_lofo_R2):
            num(f"Delta{key}", r.delta_lofo_R2, "{:+.3f}")
            num(f"CiLo{key}", r.ci95_low, "{:+.3f}"); num(f"CiHi{key}", r.ci95_high, "{:+.3f}")
            num(f"Spg{key}", r.smallest_provable_gain, "{:.2f}")
    xf = m[(m.estimator == "xgboost") & (m.arm == "conditions")].iloc[0]
    num("LofoXgbConditions", xf.lofo_R2, "{:.3f}"); num("GroupedXgbConditions", xf.grouped_R2, "{:.3f}")
    num("SpgMin", min(r.smallest_provable_gain for r in chem), "{:.2f}")
    num("SpgMax", max(r.smallest_provable_gain for r in chem), "{:.2f}")
    num("LadderRepeats", int(rows[0].n_repeats), "{}")


# --------------------------------------------------------------------------
# Figure 5 -- the formulation space as the model sees it
# --------------------------------------------------------------------------
def fig_umap():
    """The ingredient descriptions as the text encoder sees them.

    Every distinct released ingredient string is embedded with MatSciBERT (the
    encoder behind the text arm) from the production cache, keyed by the sha1
    of the string, and projected to two dimensions with UMAP. Panel (a) colors
    by taxonomy role, panel (b) shows the base polymers alone colored by the
    reactive-group tag the taxonomy pass assigned (alkenyl, silanol, Si-H, ...),
    with the most frequent descriptions annotated.
    """
    import hashlib
    import umap

    comp = pd.read_csv(SNAP / "components.csv", usecols=["formulation_id", "text", "category_labels"])
    comp = comp.dropna(subset=["text"])
    freq = comp.groupby("text")["formulation_id"].nunique()
    labels = comp.groupby("text")["category_labels"].first().fillna("")
    texts = list(freq.index)

    z = np.load(RTV / "data" / "ml_models" / "ingredient_description_embeddings__matscibert.npz",
                allow_pickle=True)
    cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keyed = [(t, cache.get(hashlib.sha1(t.encode("utf-8")).hexdigest())) for t in texts]
    have = [(t, v) for t, v in keyed if v is not None]
    n_missing = len(texts) - len(have)
    print(f"[umap] {len(texts)} distinct ingredient strings, {n_missing} without a cached vector")
    texts = [t for t, _ in have]
    X = np.stack([v for _, v in have]).astype(float)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    Z = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=0).fit_transform(X)

    lab = [labels[t].split("|") for t in texts]
    role = np.array([l[0] if l and l[0] else "unknown_role" for l in lab])
    n_form = np.array([freq[t] for t in texts])

    ROLE_GROUPS = [("base_polymer", "base polymer", BLUE), ("crosslinker", "crosslinker", ORANGE),
                   ("cure_catalyst", "cure catalyst", PURPLE), ("cure_initiator", "cure initiator", "#c62828"),
                   ("cure_inhibitor", "cure inhibitor", GOLD), ("reinforcing_filler", "reinforcing filler", TEAL),
                   ("other_filler", "other filler / treating agent", "#2e7d32")]
    filler_like = {"extending_filler", "functional_filler", "lightweight_filler", "filler_treating_agent"}
    grp = np.where(np.isin(role, list(filler_like)), "other_filler", role)
    named = {g for g, _, _ in ROLE_GROUPS}

    fig, axes = plt.subplots(1, 2, figsize=(24, 9.2), layout="constrained", gridspec_kw=dict(width_ratios=[1.0, 1.35]))
    fig.get_layout_engine().set(w_pad=0.3, wspace=0.05)

    # (a) every distinct ingredient description, colored by role
    ax = axes[0]
    other = ~np.isin(grp, list(named))
    ax.scatter(Z[other, 0], Z[other, 1], s=34, c=GREY, alpha=0.6, edgecolor="none",
               label=f"other roles ({other.sum()})")
    for g, name, c in ROLE_GROUPS:
        m = grp == g
        ax.scatter(Z[m, 0], Z[m, 1], s=30 + 6 * np.sqrt(n_form[m]), c=c, alpha=0.85, edgecolor="white", lw=0.4,
                   label=f"{name} ({m.sum()})")
    xa0, xa1 = Z[:, 0].min(), Z[:, 0].max()
    ax.set_xlim(xa0 - 0.45 * (xa1 - xa0), xa1 + 0.05 * (xa1 - xa0))
    ax.legend(loc="lower left", frameon=False, prop=dict(weight="bold", size=MIN_FONTSIZE), markerscale=1.3,
              handletextpad=0.3, labelspacing=0.3)
    style_axes(ax, xlabel="UMAP 1", ylabel="UMAP 2", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    ax.set_xticks([]); ax.set_yticks([])
    clean(ax, ""); panel(ax, "a")

    # (b) base polymers only, colored by the reactive group the taxonomy names
    ax = axes[1]
    bp = grp == "base_polymer"
    Xb = X[bp]
    Zb = umap.UMAP(n_neighbors=10, min_dist=0.4, metric="cosine", random_state=0).fit_transform(Xb)
    tb = [t for t, m in zip(texts, bp) if m]
    lb = [l for l, m in zip(lab, bp) if m]
    nb = n_form[bp]
    REACT = [("alkenyl", "vinyl / alkenyl", BLUE), ("silanol", "silanol (Si-OH)", ORANGE), ("sih", "Si-H", PURPLE),
             ("non_reactive", "no reactive unit named", GREY), ("unknown_reactivity", "reactivity not stated", "#5c6770")]
    react = np.array([next((k for k, _, _ in REACT if k in l), "unknown_reactivity") for l in lb])
    for k, name, c in REACT:
        m = react == k
        if m.any():
            ax.scatter(Zb[m, 0], Zb[m, 1], s=40 + 8 * np.sqrt(nb[m]), c=c, alpha=0.85, edgecolor="white", lw=0.5,
                       label=f"{name} ({m.sum()})")
    # annotate one exemplar per region: k-means medoids in embedding space,
    # labelled in a column to the right of the cloud with leader lines
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(Xb)
    ex = []
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) < 4:
            continue
        # the member closest to the centroid whose first clause is informative
        d = np.linalg.norm(Xb[idx] - km.cluster_centers_[c], axis=1)
        for i in idx[np.argsort(d)]:
            clause = tb[i].split(". ")[0].strip()
            if len(clause) >= 18:
                ex.append((i, clause))
                break
    ex.sort(key=lambda e: -Zb[e[0], 1])
    x0, x1 = Zb[:, 0].min(), Zb[:, 0].max()
    y0, y1 = Zb[:, 1].min(), Zb[:, 1].max()
    ax.set_xlim(x0 - 0.05 * (x1 - x0), x1 + 1.05 * (x1 - x0))
    ys = np.linspace(y1, y0, len(ex))
    for (i, clause), yy in zip(ex, ys):
        short = clause if len(clause) <= 58 else clause[:56].rstrip() + "…"
        ax.annotate(short, (Zb[i, 0], Zb[i, 1]), xytext=(x1 + 0.1 * (x1 - x0), yy), textcoords="data",
                    fontsize=MIN_FONTSIZE, fontweight="bold", color=NAVY, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.0, alpha=0.7, shrinkB=4))
    ax.legend(loc="lower left", frameon=False, prop=dict(weight="bold", size=MIN_FONTSIZE), markerscale=1.3,
              handletextpad=0.3, labelspacing=0.3)
    style_axes(ax, xlabel="UMAP 1", ylabel="UMAP 2", label_fontsize=FS_LABEL, tick_fontsize=FS_TICK)
    ax.set_xticks([]); ax.set_yticks([])
    clean(ax, ""); panel(ax, "b")

    save(fig, "umap_of_ingredient_descriptions_by_role_and_base_polymer_reactive_group")
    num("UmapStrings", len(texts)); num("UmapMissing", n_missing); num("UmapDims", X.shape[1])
    num("UmapBasePolymers", int(bp.sum()))
    for k, _, _ in REACT:
        num("React" + "".join(w.capitalize() for w in k.split("_")), int((react == k).sum()))

    # Caveat numbers for the report: state of the frozen text-embedding block.
    blocks = pd.read_csv(SNAP / "formulation_blocks.csv")
    ecols = [c for c in blocks.columns if c.startswith("emb_weighted__")]
    E = blocks[ecols].to_numpy(float)
    num("EmbZeroForms", int((np.abs(E).sum(axis=1) == 0).sum()))
    num("EmbDistinct", len(np.unique(np.round(E, 6), axis=0)))
    num("EmbForms", len(blocks))
    num("EmbStrings", len(freq))


FIGURES = {"problem": fig_problem, "pipeline": fig_pipeline, "dataset": fig_dataset, "ladder": fig_ladder,
           "umap": fig_umap}


def write_numbers() -> None:
    lines = ["% Generated by technical_report/make_figures.py -- do not edit by hand.",
             "% One macro per number the report quotes; rerun the script to refresh."]
    for k, v in sorted(NUMBERS.items()):
        lines.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"[numbers] {len(NUMBERS)} macros -> {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for name in wanted:
        FIGURES[name]()
    if not sys.argv[1:]:
        write_numbers()
    else:
        # partial runs still refresh the macros they touched
        existing = {}
        p = GEN / "numbers.tex"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("\\newcommand{\\n"):
                    k = line[len("\\newcommand{\\n"):line.index("}")]
                    existing[k] = line[line.index("}{") + 2:-1]
        # Partial runs should not preserve aliases left by an older version
        # of a figure's macro names.
        for stale in ("CompositefoamRows", "CompositeonlyRows", "FoamonlyRows"):
            existing.pop(stale, None)
        existing.update(NUMBERS)
        NUMBERS.clear(); NUMBERS.update(existing)
        write_numbers()
