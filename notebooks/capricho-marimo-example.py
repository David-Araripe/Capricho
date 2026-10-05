import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import warnings

    import altair as alt
    import marimo as mo
    import matplotlib
    import numpy as np
    import pandas as pd
    from loguru import logger as loguru_logger
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from scipy import stats

    # Keep the live demo clean. Capricho configures a loguru sink at import and
    # logs at INFO by default; inside marimo those multi-line log lines get dumped
    # as raw loguru record dicts, flooding cell outputs. Disabling the "Capricho"
    # logger stops them at the source (no emission, no dump).
    # Note: Capricho.logger.set_log_level("WARNING") does NOT work here — its
    # setup_logger() hard-codes the stderr sink at level="INFO" regardless of the
    # requested level, so INFO keeps printing. logger.disable() is the reliable fix.
    loguru_logger.disable("Capricho")

    # Silence scipy's constant-input correlation warning. We guard against constant
    # arrays ourselves, but filter it defensively so no warning reaches the screen.
    warnings.filterwarnings("ignore", message=".*input array is constant.*")
    warnings.filterwarnings("ignore", message=".*Precision loss occurred.*")

    from Capricho.analysis import (
        DroppingComment,
        build_query_string,
        explode_assay_comparability,
        get_all_comments,
        plot_multi_panel_comparability,
        plot_subset,
        r2_score,
    )
    from Capricho.core.binarization import (
        VALID_CONFLICT_STRATEGIES,
        binarize_aggregated_data,
    )
    from Capricho.core.default_fields import (
        DATA_DROPPING_COMMENT,
        DATA_PROCESSING_COMMENT,
        multiple_value_cols,
    )

    # Projector-friendly matplotlib defaults: larger fonts, clean panels.
    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Rendering/interaction caps keep the app responsive WITHOUT biasing the analysis.
    # Comparability metrics are computed on ALL multi-assay compounds by default (matching
    # the CLI / case-1 analysis). When a smaller scope is chosen it is a *random*,
    # representative sample — never a head slice of a (usually target-sorted) file, which
    # would silently distort every metric. Only the browser scatter and the binarization
    # slider work on bounded slices, and those are noted on screen.
    SCATTER_POINT_CAP = 3_000  # points drawn in the interactive scatter (metrics use all data)
    BINARIZE_PREVIEW_CAP = 5_000  # random rows binarized live per slider tick
    MOLECULE_CARD_CAP = 8  # structure cards rendered from a chart selection
    COMPARABILITY_SAMPLE_OPTIONS = {
        "All compounds (exact — matches the CLI analysis)": None,
        "Random 15,000 (fast)": 15_000,
        "Random 5,000 (fastest)": 5_000,
    }
    return (
        AllChem,
        BINARIZE_PREVIEW_CAP,
        COMPARABILITY_SAMPLE_OPTIONS,
        Chem,
        DATA_DROPPING_COMMENT,
        DATA_PROCESSING_COMMENT,
        DroppingComment,
        MOLECULE_CARD_CAP,
        SCATTER_POINT_CAP,
        VALID_CONFLICT_STRATEGIES,
        alt,
        binarize_aggregated_data,
        build_query_string,
        explode_assay_comparability,
        get_all_comments,
        io,
        mo,
        multiple_value_cols,
        np,
        pd,
        plot_multi_panel_comparability,
        plot_subset,
        r2_score,
        rdMolDraw2D,
        stats,
        warnings,
    )


@app.cell
def _(
    AllChem,
    Chem,
    DATA_DROPPING_COMMENT,
    DATA_PROCESSING_COMMENT,
    alt,
    get_all_comments,
    mo,
    multiple_value_cols,
    np,
    pd,
    r2_score,
    rdMolDraw2D,
    stats,
    warnings,
):
    def load_dataframe(source, filename):
        """Load a full DataFrame from a path or bytes, detecting format from the extension.

        Reads the whole file so the analysis sees every target (Capricho outputs are
        typically sorted by target, so a head slice would be badly non-representative).
        CSV/TSV use the pyarrow engine — it reads a 260 MB file in well under a second —
        and fall back to the default parser if pyarrow is unavailable or chokes.
        """
        name = str(filename).lower()
        if name.endswith(".parquet"):
            return pd.read_parquet(source)
        sep = "\t" if name.endswith(".tsv") else ","
        try:
            return pd.read_csv(source, sep=sep, engine="pyarrow")
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=sep, low_memory=False)

    def detect_data_properties(df):
        """Detect a loaded DataFrame's structure: aggregation, value column, flags."""
        is_aggregated = False
        for col in multiple_value_cols:
            if col in df.columns and df[col].astype(str).str.contains("|", regex=False).any():
                is_aggregated = True
                break

        value_column = "pchembl_value"
        if "pchembl_value" not in df.columns and "standard_value" in df.columns:
            value_column = "standard_value"
        is_log_scale = value_column == "pchembl_value"
        mean_col = f"{value_column}_mean"

        # Use the same axis window Capricho's plot_subset / plot_multi_panel_comparability
        # use: (3, 12) for pChEMBL, otherwise auto-detected from the values. This keeps the
        # interactive Altair scatter visually aligned with the library matplotlib panels.
        if is_log_scale:
            axis_limits = (3.0, 12.0)
        else:
            axis_limits = None
            _src_col = value_column if value_column in df.columns else mean_col
            if _src_col in df.columns:
                _sample = df[_src_col].dropna().astype(str).head(5000)
                _flat = pd.to_numeric(pd.Series("|".join(_sample).split("|")), errors="coerce").dropna()
                if len(_flat) > 0:
                    lo, hi = float(_flat.quantile(0.005)), float(_flat.quantile(0.995))
                    margin = max((hi - lo) * 0.05, 1e-9)
                    axis_limits = (lo - margin, hi + margin)

        # Most common standard_type, for figure titles (e.g. "IC50").
        assay_label = ""
        if "standard_type" in df.columns:
            _types = df["standard_type"].dropna().astype(str).str.split("|").explode().str.strip()
            _types = _types[(_types != "") & (_types.str.lower() != "nan")]
            if len(_types) > 0:
                assay_label = _types.value_counts().index[0]

        present_flags = []
        flag_cols = [c for c in (DATA_DROPPING_COMMENT, DATA_PROCESSING_COMMENT) if c in df.columns]
        joined = {c: df[c].fillna("").astype(str) for c in flag_cols}
        for flag in get_all_comments():
            if any(joined[c].str.contains(flag, regex=False).any() for c in flag_cols):
                present_flags.append(flag)

        return {
            "is_aggregated": is_aggregated,
            "value_column": value_column,
            "is_log_scale": is_log_scale,
            "axis_limits": axis_limits or (3.0, 12.0),
            "assay_label": assay_label,
            "present_flags": present_flags,
            "has_mean": mean_col in df.columns,
        }

    def honest_metrics(x, y, is_log=True, min_points=3):
        """Guarded metrics matching Capricho's library plots.

        Reports R² (Capricho's ``r2_score`` — the exact value annotated on
        ``plot_subset`` / ``plot_multi_panel_comparability``), Spearman rho and
        Kendall tau, plus the fraction of pairs within +/-0.3 and +/-1.0 for
        log-scale (pChEMBL) data. Requires >= ``min_points`` finite, non-constant
        points; otherwise every metric is None so the caller shows "n/a" rather
        than the blow-up ``r2_score`` produces on a near-degenerate subset.
        """
        x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        out = {"n": n, "r2": None, "spearman": None, "kendall": None, "within_03": None, "within_10": None}
        if n < min_points or np.ptp(x) == 0 or np.ptp(y) == 0:
            return out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2 = r2_score(x, y)
            spear = stats.spearmanr(x, y)[0]
            ken = stats.kendalltau(x, y)[0]
        out["r2"] = float(r2) if np.isfinite(r2) and abs(r2) < 1e3 else None
        out["spearman"] = float(spear) if np.isfinite(spear) else None
        out["kendall"] = float(ken) if np.isfinite(ken) else None
        if is_log:
            d = np.abs(x - y)
            out["within_03"] = float(np.mean(d <= 0.3))
            out["within_10"] = float(np.mean(d <= 1.0))
        return out

    def _fmt(v, pct=False):
        if v is None or not np.isfinite(v):
            return "n/a"
        return f"{v * 100:.1f}%" if pct else f"{v:.3f}"

    def metrics_md(m, is_log=True):
        """Render an honest-metrics dict as a compact markdown block."""
        if m["n"] < 3 or m["r2"] is None:
            return mo.md(f"**Agreement metrics** — n/a *(only {m['n']:,} usable pair(s))*")
        lines = [
            "**Agreement metrics**",
            "",
            "| Pairs | R² | Spearman ρ | Kendall τ |",
            "|---|---|---|---|",
            f"| {m['n']:,} | {_fmt(m['r2'])} | {_fmt(m['spearman'])} | {_fmt(m['kendall'])} |",
        ]
        if is_log and m["within_10"] is not None:
            lines += [
                "",
                f"Within ±0.3 log units: **{_fmt(m['within_03'], pct=True)}** · "
                f"within ±1.0 log units: **{_fmt(m['within_10'], pct=True)}**",
            ]
        return mo.md("\n".join(lines))

    def comparability_scatter(data, title, color, value_column, axis_limits, is_log, point_cap):
        """Interactive Altair scatter with identity and +/-0.3 / +/-1.0 reference lines.

        Down-samples to ``point_cap`` points for a responsive browser; a caption
        notes when sampling was applied. Metrics are computed elsewhere on all data.
        """
        x_col, y_col = f"{value_column}_x", f"{value_column}_y"
        lo, hi = axis_limits
        plot_df = data.reset_index(drop=True)
        note = ""
        if len(plot_df) > point_cap:
            plot_df = plot_df.sample(n=point_cap, random_state=0).reset_index(drop=True)
            note = f"Showing a random {point_cap:,} of {len(data):,} points."

        # clip (not clamp) points and lines to the axis box, exactly like the matplotlib
        # routine: out-of-range points are dropped and the reference lines stay straight
        # and parallel, instead of clamp bending them toward identity (which reads as a fit).
        base = (
            alt.Chart(plot_df)
            .mark_circle(size=70, opacity=0.5, color=color, clip=True)
            .encode(
                x=alt.X(f"{x_col}:Q", title="Assay 1", scale=alt.Scale(domain=[lo, hi])),
                y=alt.Y(f"{y_col}:Q", title="Assay 2", scale=alt.Scale(domain=[lo, hi])),
                tooltip=[
                    alt.Tooltip("connectivity:N", title="Compound"),
                    alt.Tooltip(f"{x_col}:Q", title="Assay 1", format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", title="Assay 2", format=".2f"),
                    alt.Tooltip("assay_chembl_id_x:N", title="Assay 1 ID"),
                    alt.Tooltip("assay_chembl_id_y:N", title="Assay 2 ID"),
                ],
            )
        )
        ref_specs = [(0.0, "black", [0])]
        if is_log:
            ref_specs += [
                (1.0, "black", [6, 4]),
                (-1.0, "black", [6, 4]),
                (0.3, "gray", [2, 3]),
                (-0.3, "gray", [2, 3]),
            ]
        layers = [base]
        for offset, col, dash in ref_specs:
            line_df = pd.DataFrame({"x": [lo, hi], "y": [lo + offset, hi + offset]})
            layers.append(
                alt.Chart(line_df)
                .mark_line(color=col, strokeDash=dash, opacity=0.7, clip=True)
                .encode(x="x:Q", y="y:Q")
            )
        chart = alt.layer(*layers).properties(width=460, height=460, title=title)
        return chart, note

    def _svg(smiles, size=190):
        if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)) or str(smiles) in ("", "nan"):
            return ""
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def _assay_link(assay_id):
        aid = str(assay_id)
        return (
            f'<a href="https://www.ebi.ac.uk/chembl/explore/assay/{aid}" target="_blank" '
            f'style="color:#2563eb;text-decoration:none;font-weight:600;">{aid} ↗</a>'
        )

    def molecule_cards(rows, value_column, is_log, max_cards):
        """Build a grid of structure cards for selected pairwise comparisons.

        Each card shows the two co-crystallised assay measurements of the same
        compound side by side (rendered SVG, value, assay link) and their delta.
        Returns a single mo.Html blob so the inline SVGs render as pictures.
        """
        x_col, y_col = f"{value_column}_x", f"{value_column}_y"
        subset = rows.head(max_cards)
        cards = []
        for _, r in subset.iterrows():
            sx = r.get("canonical_smiles_x", "")
            sy = r.get("canonical_smiles_y", "")
            try:
                vx = float(r[x_col])
                vy = float(r[y_col])
                delta = abs(vx - vy)
                vx_s, vy_s, delta_s = f"{vx:.2f}", f"{vy:.2f}", f"{delta:.2f}"
            except (TypeError, ValueError):
                vx_s = vy_s = delta_s = "—"
                delta = 0.0
            unit = " log units" if is_log else ""
            badge_color = (
                "#dc2626"
                if (is_log and delta > 1.0)
                else ("#d97706" if (is_log and delta > 0.3) else "#16a34a")
            )
            conn = str(r.get("connectivity", ""))[:22]

            def _pane(smiles, val, assay_col, row=r):
                aid = row.get(assay_col, "")
                return (
                    '<div style="text-align:center;flex:1;">'
                    f'<div style="background:#fff;border-radius:8px;padding:4px;">{_svg(smiles)}</div>'
                    f'<div style="font-size:15px;font-weight:700;margin-top:4px;">{val}</div>'
                    f'<div style="font-size:12px;margin-top:2px;">{_assay_link(aid)}</div>'
                    "</div>"
                )

            cards.append(
                '<div style="border:1px solid #d1d5db;border-radius:12px;padding:12px;margin:6px;'
                'width:430px;box-shadow:0 1px 3px rgba(0,0,0,0.08);background:#f9fafb;">'
                f'<div style="font-size:12px;color:#6b7280;margin-bottom:6px;">Compound <code>{conn}</code></div>'
                '<div style="display:flex;align-items:flex-start;gap:8px;">'
                f"{_pane(sx, vx_s, 'assay_chembl_id_x')}"
                f'<div style="align-self:center;text-align:center;color:{badge_color};font-weight:700;font-size:13px;">'
                f'Δ {delta_s}<br><span style="font-size:11px;font-weight:400;color:#6b7280;">{unit.strip()}</span></div>'
                f"{_pane(sy, vy_s, 'assay_chembl_id_y')}"
                "</div></div>"
            )
        grid = (
            '<div style="display:flex;flex-wrap:wrap;justify-content:flex-start;">'
            + "".join(cards)
            + "</div>"
        )
        return mo.Html(grid)

    return (
        comparability_scatter,
        detect_data_properties,
        honest_metrics,
        load_dataframe,
        metrics_md,
        molecule_cards,
    )


@app.cell
def _(COMPARABILITY_SAMPLE_OPTIONS, mo):
    file_upload = mo.ui.file(
        filetypes=[".csv", ".tsv", ".parquet"],
        kind="area",
        label="Drag & drop a file",
    )
    file_browser = mo.ui.file_browser(
        filetypes=[".csv", ".tsv", ".parquet"],
        multiple=False,
        label="…or browse to one",
    )
    comparability_scope = mo.ui.dropdown(
        options=list(COMPARABILITY_SAMPLE_OPTIONS.keys()),
        value=next(iter(COMPARABILITY_SAMPLE_OPTIONS)),
        label="Comparability scope",
    )
    cli_file_path = mo.cli_args().get("file", None)

    mo.vstack(
        [
            mo.md("# 🧪 CAPRICHO Data Explorer"),
            mo.md(
                "Interactively explore how each **data-quality flag** affects the comparability "
                "of aggregated ChEMBL bioactivity data — then clean and binarize it. "
                "Load a Capricho output file (CSV, TSV, or Parquet) to begin."
            ),
            mo.hstack([file_upload, file_browser], justify="start", gap=2, widths=[1, 1]),
            comparability_scope,
        ]
    )
    return cli_file_path, comparability_scope, file_browser, file_upload


@app.cell
def _(cli_file_path, file_browser, file_upload, io, load_dataframe, mo):
    _df = None
    _name = None

    if cli_file_path is not None:
        _name = str(cli_file_path)
        _df = load_dataframe(_name, _name)
    elif file_browser.value and len(file_browser.value) > 0:
        _name = file_browser.name()
        _df = load_dataframe(file_browser.path(), _name)
    elif file_upload.value and len(file_upload.value) > 0:
        _name = file_upload.name()
        _df = load_dataframe(io.BytesIO(file_upload.contents()), _name)

    mo.stop(
        _df is None,
        mo.callout(
            mo.md("⬆️ **Upload or browse to a Capricho data file to begin.**"),
            kind="info",
        ),
    )

    loaded_df = _df
    source_filename = _name

    mo.callout(
        mo.md(f"✅ Loaded **{source_filename.split('/')[-1]}** — **{len(loaded_df):,}** rows (full file)"),
        kind="success",
    )
    return loaded_df, source_filename


@app.cell
def _(detect_data_properties, loaded_df, mo, source_filename):
    data_props = detect_data_properties(loaded_df)

    _flags = data_props["present_flags"]
    _flags_str = "<br>".join(f"• {f}" for f in _flags) if _flags else "*none detected*"
    _agg = (
        "✅ Yes (pipe-separated measurements)"
        if data_props["is_aggregated"]
        else "❌ No (single value per row)"
    )
    _lo, _hi = data_props["axis_limits"]

    mo.vstack(
        [
            mo.md("## 1 · Dataset overview"),
            mo.hstack(
                [
                    mo.stat(label="Rows", value=f"{len(loaded_df):,}"),
                    mo.stat(label="Columns", value=f"{len(loaded_df.columns)}"),
                    mo.stat(label="Value column", value=data_props["value_column"]),
                    mo.stat(label="Flags present", value=f"{len(_flags)}"),
                ],
                justify="start",
                gap=1,
            ),
            mo.md(f"""
    | Property | Value |
    |----------|-------|
    | **File** | `{source_filename.split('/')[-1]}` |
    | **Aggregated** | {_agg} |
    | **Scale** | {"pChEMBL (log units)" if data_props["is_log_scale"] else "raw standard_value"} |
    | **Axis range** | {_lo:.1f} – {_hi:.1f} |
    | **Quality / processing flags** | {_flags_str} |
    """),
        ]
    )
    return (data_props,)


@app.cell
def _(
    COMPARABILITY_SAMPLE_OPTIONS,
    comparability_scope,
    data_props,
    explode_assay_comparability,
    loaded_df,
    mo,
):
    mo.stop(
        not data_props["is_aggregated"],
        mo.callout(
            mo.md(
                "This file is **not aggregated**, so there are no repeated measurements to compare. "
                "Comparability, cleaning and binarization need aggregated output from `capricho get`."
            ),
            kind="warn",
        ),
    )

    _val = data_props["value_column"]
    _multi = loaded_df.query(f'{_val}.str.contains("|", regex=False)')

    mo.stop(
        len(_multi) == 0,
        mo.callout(
            mo.md("No compounds with **multiple assay measurements** found in this file."), kind="warn"
        ),
    )

    # Use every multi-assay compound by default (exact, matches the CLI analysis). If a
    # smaller scope is chosen, draw a *random* representative sample — a head slice of a
    # target-sorted file would cover only a handful of targets and distort every metric.
    _total_multi = len(_multi)
    _sample_n = COMPARABILITY_SAMPLE_OPTIONS.get(comparability_scope.value)
    _mol_note = ""
    if _sample_n is not None and _total_multi > _sample_n:
        _multi = _multi.sample(n=_sample_n, random_state=0)
        _mol_note = (
            f" *(random representative sample of {_sample_n:,} of {_total_multi:,} multi-assay "
            f"compounds — choose “All compounds” for exact results)*"
        )

    comparability_subset = _multi.assign(repeat=lambda x: range(len(x)))
    exploded_subset = explode_assay_comparability(comparability_subset, value_column=_val)
    for _c in (f"{_val}_x", f"{_val}_y"):
        exploded_subset[_c] = exploded_subset[_c].astype(float)

    mo.md(
        f"## 2 · Assay comparability\n\n"
        f"**{len(comparability_subset):,}** compounds measured in multiple assays "
        f"→ **{len(exploded_subset):,}** pairwise assay comparisons{_mol_note}. "
        f"Each point below is one compound measured in two different assays; points on the diagonal agree."
    )
    return (exploded_subset,)


@app.cell
def _(
    data_props,
    exploded_subset,
    get_all_comments,
    mo,
    plot_multi_panel_comparability,
):
    _label = data_props["assay_label"] or "Assay"
    _fig, _axs = plot_multi_panel_comparability(
        exploded_subset,
        get_all_comments(),
        title=f"{_label} Comparability Across Flagged Data",
        figsize=(22, 8),
        ncols=5,
        value_column=data_props["value_column"],
    )

    _caption = (
        "**Comparability across every data-quality flag** — Capricho's "
        "`plot_multi_panel_comparability`. Solid line = identity (y=x); dashed = y=x±1; "
        "dash-dot = y=x±0.3. Each panel is annotated with R², Spearman ρ and Kendall τ. "
        "Flags whose scatter spreads far off the diagonal mark measurements that are hard "
        "to compare across assays."
    )
    mo.vstack([mo.md("### Overview — all flags at a glance"), _fig, mo.md(_caption)])
    return


@app.cell
def _(data_props, mo):
    _flags = data_props["present_flags"]
    flag_selector = mo.ui.dropdown(
        options=_flags,
        value=_flags[0] if _flags else None,
        label="Inspect one flag",
        full_width=True,
    )
    mo.vstack([mo.md("### Inspect a single flag"), flag_selector])
    return (flag_selector,)


@app.cell
def _(
    SCATTER_POINT_CAP,
    build_query_string,
    comparability_scatter,
    data_props,
    exploded_subset,
    flag_selector,
    mo,
):
    mo.stop(flag_selector.value is None, mo.md("*No flags to inspect.*"))

    _val = data_props["value_column"]
    _flag_data = exploded_subset.query(build_query_string(flag_selector.value, value_column=_val))

    mo.stop(
        len(_flag_data) == 0,
        mo.callout(mo.md(f"No pairwise comparisons carry the flag **{flag_selector.value}**."), kind="info"),
    )

    _keep = [
        c
        for c in (
            f"{_val}_x",
            f"{_val}_y",
            "connectivity",
            "assay_chembl_id_x",
            "assay_chembl_id_y",
            "canonical_smiles_x",
            "canonical_smiles_y",
        )
        if c in _flag_data.columns
    ]
    flag_data = _flag_data[_keep].copy()

    _chart, _note = comparability_scatter(
        flag_data,
        title=f"{flag_selector.value}",
        color="#2563eb",
        value_column=_val,
        axis_limits=data_props["axis_limits"],
        is_log=data_props["is_log_scale"],
        point_cap=SCATTER_POINT_CAP,
    )
    flag_chart = mo.ui.altair_chart(_chart)

    mo.vstack(
        [
            flag_chart,
            mo.md(f"*Drag a box on the plot to select points and inspect their structures below.* {_note}"),
        ]
    )
    return flag_chart, flag_data


@app.cell
def _(data_props, flag_data, honest_metrics, metrics_md):
    _val = data_props["value_column"]
    _m = honest_metrics(flag_data[f"{_val}_x"], flag_data[f"{_val}_y"], is_log=data_props["is_log_scale"])
    metrics_md(_m, is_log=data_props["is_log_scale"])
    return


@app.cell
def _(
    MOLECULE_CARD_CAP,
    data_props,
    flag_chart,
    flag_data,
    mo,
    molecule_cards,
):
    # The scatter is a layered chart, so `flag_chart.value` is unavailable; filter
    # our own DataFrame with `apply_selection`. That returns the *full* frame when
    # nothing is selected, so first check for a real (non-pan/zoom) selection.
    _sel_dict = flag_chart.selections or {}
    _has_selection = any(not str(k).startswith("pan_zoom") for k in _sel_dict)
    _sel = flag_chart.apply_selection(flag_data) if _has_selection else None

    if _sel is None or len(_sel) == 0:
        mo.output.replace(
            mo.callout(
                mo.md(
                    "🔍 *Drag a selection box on the scatter above to see the compounds and their two assays here.*"
                ),
                kind="neutral",
            )
        )
    else:
        _val = data_props["value_column"]
        _n = len(_sel)
        _head = mo.md(
            f"**{_n:,} point(s) selected.**"
            + (f" Showing the first {MOLECULE_CARD_CAP}." if _n > MOLECULE_CARD_CAP else "")
        )
        _cards = molecule_cards(
            _sel,
            value_column=_val,
            is_log=data_props["is_log_scale"],
            max_cards=MOLECULE_CARD_CAP,
        )
        mo.output.replace(mo.vstack([_head, _cards]))
    return


@app.cell
def _(DroppingComment, data_props, mo):
    mo.stop(not data_props["is_aggregated"])

    _dropping = {dc.value for dc in DroppingComment}
    dropping_flags_present = [
        f for f in data_props["present_flags"] if any(f.startswith(dv) for dv in _dropping)
    ]

    flag_multiselect = mo.ui.multiselect(
        options=dropping_flags_present,
        label="Quality flags to exclude",
        full_width=True,
    )
    dedup_checkbox = mo.ui.checkbox(label="Exclude cross-document duplicate pairs (assay 1 = assay 2)")

    mo.vstack(
        [
            mo.md(
                "## 3 · Cleaning\n\n"
                "CAPRICHO never silently drops data. Choose which flagged comparisons to exclude and watch "
                "comparability improve — the same flag-filtering the case-1 notebook applies to its "
                "*unprocessed* → *cleaned* scatters. Pairs where either assay carries a selected flag are removed."
            ),
            (
                flag_multiselect
                if dropping_flags_present
                else mo.md("*No dropping flags present in this dataset.*")
            ),
            dedup_checkbox,
        ]
    )
    return dedup_checkbox, flag_multiselect


@app.cell
def _(data_props, dedup_checkbox, exploded_subset, flag_multiselect, mo):
    _val = data_props["value_column"]
    _selected = list(flag_multiselect.value)
    _dedup = dedup_checkbox.value
    cleaning_active = bool(_selected) or _dedup

    # Case-1 approach: filter the already-exploded pairwise comparisons, dropping pairs
    # whose combined dropping/processing comment carries a selected flag. This is a plain
    # pandas query on the pre-computed frame, so it stays instant on the full dataset.
    cleaned_exploded = exploded_subset
    if cleaning_active:
        _flag_pattern = "|".join(_selected)
        _parts = []
        if _selected:
            _parts.append("~dropping_comment.str.contains(@_flag_pattern, regex=True)")
        if _dedup:
            _parts.append(f"{_val}_x != {_val}_y")
        cleaned_exploded = exploded_subset.query(" & ".join(_parts))

    _before, _after = len(exploded_subset), len(cleaned_exploded)
    _retained = _after / max(_before, 1) * 100
    mo.callout(
        mo.md(
            f"**Pairwise comparisons:** {_before:,} → **{_after:,}** ({_retained:.1f}% retained) · "
            f"**excluded flags:** {', '.join(_selected) if _selected else 'none'} · "
            f"**duplicate pairs removed:** {'yes' if _dedup else 'no'}"
        ),
        kind="info" if not cleaning_active else "success",
    )
    return cleaned_exploded, cleaning_active


@app.cell
def _(
    cleaned_exploded,
    cleaning_active,
    data_props,
    exploded_subset,
    honest_metrics,
    mo,
    plot_subset,
):
    mo.stop(
        not cleaning_active,
        mo.md("*Select flags to exclude or enable deduplication above to see the before/after comparison.*"),
    )

    _val = data_props["value_column"]
    mo.stop(len(cleaned_exploded) == 0, mo.md("*No comparisons remain after cleaning.*"))

    # Same plot_subset the case-1 notebook uses for its unprocessed/cleaned scatters.
    _before_fig, _ = plot_subset(
        exploded_subset, title="Before cleaning", color="slategray", figsize=(5, 4.6), value_column=_val
    )
    _after_fig, _ = plot_subset(
        cleaned_exploded, title="After cleaning", color="forestgreen", figsize=(5, 4.6), value_column=_val
    )

    _mb = honest_metrics(
        exploded_subset[f"{_val}_x"], exploded_subset[f"{_val}_y"], data_props["is_log_scale"]
    )
    _ma = honest_metrics(
        cleaned_exploded[f"{_val}_x"], cleaned_exploded[f"{_val}_y"], data_props["is_log_scale"]
    )

    def _row(name, before, after, pct=False):
        def f(v):
            if v is None:
                return "n/a"
            return f"{v * 100:.1f}%" if pct else f"{v:.3f}"

        def d():
            if before is None or after is None:
                return "—"
            return (
                f"{(after - before) * (100 if pct else 1):+.1f}{'pp' if pct else ''}"
                if pct
                else f"{after - before:+.3f}"
            )

        return f"| **{name}** | {f(before)} | {f(after)} | {d()} |"

    _table = "\n".join(
        [
            "| Metric | Before | After | Δ |",
            "|--------|--------|-------|---|",
            _row("R²", _mb["r2"], _ma["r2"]),
            _row("Spearman ρ", _mb["spearman"], _ma["spearman"]),
            _row("Kendall τ", _mb["kendall"], _ma["kendall"]),
            (
                _row("Within ±0.3", _mb["within_03"], _ma["within_03"], pct=True)
                if data_props["is_log_scale"]
                else ""
            ),
            (
                _row("Within ±1.0", _mb["within_10"], _ma["within_10"], pct=True)
                if data_props["is_log_scale"]
                else ""
            ),
            f"| **Pairs** | {_mb['n']:,} | {_ma['n']:,} | {_ma['n'] - _mb['n']:+,} |",
        ]
    )

    mo.vstack(
        [
            mo.md("### Before vs after cleaning"),
            mo.hstack([_before_fig, _after_fig], justify="center"),
            mo.md(_table),
        ]
    )
    return


@app.cell
def _(VALID_CONFLICT_STRATEGIES, data_props, loaded_df, mo, pd):
    _val = data_props["value_column"]
    _mean_col = f"{_val}_mean"

    if _mean_col in loaded_df.columns:
        _vals = pd.to_numeric(loaded_df[_mean_col], errors="coerce").dropna()
        _lo = max(float(_vals.min()) - 0.5, 0.0) if len(_vals) else 3.0
        _hi = float(_vals.max()) + 0.5 if len(_vals) else 12.0
    else:
        _lo, _hi = 3.0, 12.0

    _default = 6.0 if (data_props["is_log_scale"] and _lo <= 6.0 <= _hi) else round((_lo + _hi) / 2, 1)

    threshold_slider = mo.ui.slider(
        start=round(_lo, 1),
        stop=round(_hi, 1),
        value=_default,
        step=0.1,
        label="Activity threshold",
        full_width=True,
        show_value=True,
    )
    conflict_selector = mo.ui.dropdown(
        options=["None", *sorted(VALID_CONFLICT_STRATEGIES)],
        value="None",
        label="Conflict resolution",
    )

    mo.vstack(
        [
            mo.md(
                "## 4 · Binarization\n\n"
                "Threshold continuous activity into **active / inactive** labels. Censored measurements "
                "(`<`, `>`) and conflicting duplicates are handled by the chosen conflict-resolution strategy."
            ),
            mo.hstack([threshold_slider, conflict_selector], justify="start", gap=2, widths=[2, 1]),
        ]
    )
    return conflict_selector, threshold_slider


@app.cell
def _(
    BINARIZE_PREVIEW_CAP,
    alt,
    binarize_aggregated_data,
    conflict_selector,
    data_props,
    loaded_df,
    mo,
    pd,
    threshold_slider,
):
    _val = data_props["value_column"]
    _mean_col = f"{_val}_mean"

    mo.stop(
        _mean_col not in loaded_df.columns,
        mo.callout(
            mo.md(f"Binarization needs a `{_mean_col}` column (aggregated output from `capricho get`)."),
            kind="warn",
        ),
    )

    _preview = loaded_df
    _preview_note = ""
    if len(loaded_df) > BINARIZE_PREVIEW_CAP:
        _preview = loaded_df.sample(n=BINARIZE_PREVIEW_CAP, random_state=0)
        _preview_note = (
            f" · live preview on a random representative {BINARIZE_PREVIEW_CAP:,} of "
            f"{len(loaded_df):,} rows (keeps the slider snappy)"
        )

    _strategy = None if conflict_selector.value == "None" else conflict_selector.value
    _bin = binarize_aggregated_data(
        _preview.copy(),
        threshold=threshold_slider.value,
        value_column=_mean_col,
        conflict_resolution=_strategy,
    )

    _active = int((_bin["activity_binary"] == 1).sum())
    _inactive = int((_bin["activity_binary"] == 0).sum())
    _unclassified = int(_bin["activity_binary"].isna().sum())
    _total = max(len(_bin), 1)

    _dist = pd.DataFrame(
        {
            "Label": ["Active (1)", "Inactive (0)", "Unclassified"],
            "Count": [_active, _inactive, _unclassified],
        }
    )
    _order = ["Active (1)", "Inactive (0)", "Unclassified"]
    _bar = (
        alt.Chart(_dist)
        .mark_bar(size=90)
        .encode(
            x=alt.X("Label:N", title=None, sort=_order),
            y=alt.Y("Count:Q", title="Compounds"),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=_order, range=["#16a34a", "#dc2626", "#9ca3af"]),
                legend=None,
            ),
            tooltip=["Label:N", "Count:Q"],
        )
        .properties(width=420, height=320, title=f"Threshold = {threshold_slider.value:.1f}")
    )

    mo.vstack(
        [
            mo.hstack([mo.ui.altair_chart(_bar)], justify="center"),
            mo.md(f"""
    | Label | Count | % |
    |-------|-------|---|
    | 🟢 **Active** | {_active:,} | {_active / _total * 100:.1f}% |
    | 🔴 **Inactive** | {_inactive:,} | {_inactive / _total * 100:.1f}% |
    | ⚪ **Unclassified** | {_unclassified:,} | {_unclassified / _total * 100:.1f}% |

    *Threshold = {threshold_slider.value:.1f} · conflict resolution = {conflict_selector.value}{_preview_note}*
    """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
