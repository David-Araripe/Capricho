"""Account for how much data carries each quality flag and how much is omitted when flags are dropped.

CAPRICHO flags data rather than silently discarding it, so the amount of data a user
actually loses depends on which flags they choose to drop. These helpers separate the
two questions:

- *flagged*: how many entries carry each flag, reported per flag for transparency;
- *omitted*: how many entries carry at least one of the flags being dropped.

The second is not the sum of the first. Flags co-occur — a measurement can be flagged
for a small assay and undefined stereochemistry at once — so summing per-flag counts
overcounts the loss. The ``Any flag`` total is the union and is the number to quote as
"data omitted".

Quality-flag percentages are relative to entries in the DataFrame passed in, so the caller
chooses the denominator: retrieved measurements, aggregated datapoints, or pairwise cross-assay
comparisons. Coverage summaries additionally report unique assay identifiers represented in an
aggregated DataFrame as their own explicitly labelled denominator.
"""

from typing import Optional, Union

import pandas as pd

from .core.default_fields import ASSAY_ID, DATA_DROPPING_COMMENT

#: Separator joining several flags written onto the same measurement (see ``add_comment``).
FLAG_SEPARATOR = " & "

ANY_FLAG_LABEL = "Any flag"
ANY_SELECTED_LABEL = "Any of the selected flags"
UNFLAGGED_LABEL = "Unflagged"
RETAINED_LABEL = "Retained"
TOTAL_LABEL = "All entries"

#: Levels at which a dataset is counted, from raw retrieval to the comparisons plotted.
RETRIEVED_LEVEL = "retrieved measurements"
AGGREGATED_LEVEL = "aggregated datapoints"
COMPARISON_LEVEL = "pairwise comparisons"
ASSAY_LEVEL = "assay identifiers represented after aggregation"

DATAPOINT_OVERLAP_LABEL = "Measured in >1 assay"
ASSAY_OVERLAP_LABEL = "Has cross-assay overlap"
NO_ASSAY_OVERLAP_LABEL = "No cross-assay overlap"


def measurement_comments(cell, sep_str: str = "|") -> list[str]:
    """Split one comment cell into the comment of each measurement it pools.

    Aggregated rows hold one comment per pooled measurement, separated by ``sep_str``.
    Non-aggregated rows yield a single comment.

    Args:
        cell: Value of a comment column for one row. Missing values yield one empty comment.
        sep_str: Separator delimiting pooled measurements.

    Returns:
        List of per-measurement comment strings, always at least one element long.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return [""]
    text = str(cell)
    if text == "nan":
        return [""]
    return text.split(sep_str)


def flags_in_comment(comment: str) -> set[str]:
    """Extract the normalized flag names present in a single measurement's comment.

    Flags carrying a run-specific value are normalized to their pattern, so that
    "Assay size < 5" and "Assay size < 20" are counted as the same reason.

    Args:
        comment: Comment of a single measurement, possibly joining several flags.

    Returns:
        Set of normalized flag names. Empty for an empty or missing comment.
    """
    from .analysis import normalize_comment_pattern

    if not comment or comment == "nan":
        return set()
    return {normalize_comment_pattern(flag.strip()) for flag in comment.split(FLAG_SEPARATOR) if flag.strip()}


def _entry_flags(
    data: pd.DataFrame,
    comment_column: str,
    sep_str: str,
    per_measurement: bool,
) -> list[set[str]]:
    """Reduce a comment column to one set of normalized flags per counted entry."""
    entries: list[set[str]] = []
    for cell in data[comment_column]:
        comments = measurement_comments(cell, sep_str)
        if per_measurement:
            entries.extend(flags_in_comment(comment) for comment in comments)
        else:
            row_flags: set[str] = set()
            for comment in comments:
                row_flags |= flags_in_comment(comment)
            entries.append(row_flags)
    return entries


def summarize_flags(
    data: pd.DataFrame,
    flags: Optional[list[str]] = None,
    comment_column: str = DATA_DROPPING_COMMENT,
    sep_str: str = "|",
    per_measurement: bool = False,
) -> pd.DataFrame:
    """Count how many entries carry each quality flag, in absolute numbers and as a share.

    Args:
        data: DataFrame holding a comment column. May be non-aggregated (one row per
            measurement), aggregated (comments pooled with ``sep_str``), or exploded
            into pairwise comparisons.
        flags: Flags to account for, normally the ones being dropped. Their union is
            reported as the data omitted. If None, every flag found in the data is
            reported and the union covers all of them.
        comment_column: Column holding the flags. Defaults to ``data_dropping_comment``;
            pass ``"dropping_comment"`` for data exploded by
            :func:`~Capricho.analysis.explode_assay_comparability`.
        sep_str: Separator delimiting pooled measurements within an aggregated cell.
        per_measurement: Count each pooled measurement separately instead of counting
            rows. Meaningful for aggregated data, where one row can pool many
            measurements. Leave False for pairwise comparisons, whose unit is the pair.

    Returns:
        DataFrame with columns ``flag``, ``kind``, ``n`` and ``pct``, sorted by
        descending count. Rows with ``kind == "flag"`` are the per-flag counts; the
        three ``kind == "total"`` rows are the union of the reported flags, the entries
        carrying none of them, and the denominator. The union is labelled
        ``"Any of the selected flags"`` and its complement ``"Retained"`` when ``flags``
        is given, describing a filter; without a selection they are ``"Any flag"`` and
        ``"Unflagged"``, describing the data as it stands.

    Raises:
        ValueError: If ``comment_column`` is not a column of ``data``.
    """
    from .analysis import normalize_comment_pattern

    if comment_column not in data.columns:
        raise ValueError(f"Column '{comment_column}' not found in DataFrame. Columns: {list(data.columns)}")

    entries = _entry_flags(data, comment_column, sep_str, per_measurement)
    n_total = len(entries)

    if flags is None:
        names = sorted({flag for entry in entries for flag in entry})
        union_label, complement_label = ANY_FLAG_LABEL, UNFLAGGED_LABEL
    else:
        names = list(dict.fromkeys(normalize_comment_pattern(str(flag)) for flag in flags))
        union_label, complement_label = ANY_SELECTED_LABEL, RETAINED_LABEL
    selected = set(names)

    counts = {name: sum(1 for entry in entries if name in entry) for name in names}
    n_omitted = sum(1 for entry in entries if entry & selected)

    def _pct(count: int) -> float:
        return count / n_total * 100 if n_total else 0.0

    rows = [
        {"flag": name, "kind": "flag", "n": counts[name], "pct": _pct(counts[name])}
        for name in sorted(names, key=lambda name: (-counts[name], name))
    ]
    rows += [
        {"flag": union_label, "kind": "total", "n": n_omitted, "pct": _pct(n_omitted)},
        {
            "flag": complement_label,
            "kind": "total",
            "n": n_total - n_omitted,
            "pct": _pct(n_total - n_omitted),
        },
        {"flag": TOTAL_LABEL, "kind": "total", "n": n_total, "pct": 100.0 if n_total else 0.0},
    ]
    return pd.DataFrame(rows, columns=["flag", "kind", "n", "pct"])


def _label_width(summary: pd.DataFrame) -> int:
    """Column width that fits every flag label in a summary."""
    return max(max((len(str(flag)) for flag in summary["flag"]), default=0) + 1, 30)


def _flag_row(flag: str, n: int, pct: float, width: int, indent: str) -> str:
    """Render one ``label: count (share)`` line, aligned to ``width``."""
    return f"{indent}{flag + ':':<{width}s} {n:>8,}  ({pct:5.1f}%)"


def format_flag_summary(
    summary: pd.DataFrame,
    title: str = "QUALITY FLAGS",
    indent: str = "  ",
    empty_message: str = "(none)",
) -> str:
    """Render a flag summary as an aligned text block for logging or printing.

    Args:
        summary: Output of :func:`summarize_flags`.
        title: Heading placed above the table.
        indent: Prefix for the heading; table rows are indented one level deeper.
        empty_message: Line shown in place of the table when no flag was found.

    Returns:
        Multi-line string, without a trailing newline.
    """
    per_flag = summary[summary["kind"] == "flag"]
    totals = summary[summary["kind"] == "total"]

    lines = [f"{indent}{title}"]
    row_indent = indent * 2
    width = _label_width(summary)

    if per_flag.empty:
        lines.append(f"{row_indent}{empty_message}")
    for row in per_flag.itertuples():
        lines.append(_flag_row(row.flag, row.n, row.pct, width, row_indent))
    lines.append(f"{row_indent}{'-' * (width + 20)}")
    for row in totals.itertuples():
        lines.append(_flag_row(row.flag, row.n, row.pct, width, row_indent))
    return "\n".join(lines)


def summarize_cross_assay_coverage(
    data: pd.DataFrame,
    assay_column: str = "assay_chembl_id",
    sep_str: str = "|",
    n_retrieved: Optional[int] = None,
) -> dict[str, Union[int, float]]:
    """Measure how much of an aggregated dataset cross-assay comparability can reach.

    Only compounds measured in more than one assay contribute a comparison, so this is
    the share of the data the comparability analysis actually observes.

    Args:
        data: Aggregated DataFrame, one row per compound-target readout.
        assay_column: Column holding the pooled assay identifiers.
        sep_str: Separator delimiting pooled measurements.
        n_retrieved: Number of measurements retrieved before aggregation, if known.
            Adds the share of retrieved measurements that the comparable rows pool.

    Returns:
        Dict with two complementary views of overlap: the number and share of aggregated
        datapoints measured in more than one assay, and the number and share of represented
        assay identifiers that co-occur with another assay in at least one datapoint. It
        also reports the number of measurements pooled by comparable datapoints and — when
        ``n_retrieved`` is given — that count as a percentage of retrieved measurements.

    Raises:
        ValueError: If ``assay_column`` is not a column of ``data``.
    """
    if assay_column not in data.columns:
        raise ValueError(f"Column '{assay_column}' not found in DataFrame. Columns: {list(data.columns)}")

    assay_values = data[assay_column].map(lambda cell: measurement_comments(cell, sep_str))
    assay_sets = assay_values.map(lambda values: {value for value in values if value and value != "nan"})
    multi_assay = assay_sets.map(len) > 1
    represented_assays = set().union(*assay_sets) if len(assay_sets) else set()
    overlapping_assays = set().union(*assay_sets[multi_assay]) if multi_assay.any() else set()

    n_datapoints = len(data)
    n_comparable = int(multi_assay.sum())
    n_assays = len(represented_assays)
    n_overlapping_assays = len(overlapping_assays)

    coverage = {
        "aggregated_datapoints": n_datapoints,
        "comparable_datapoints": n_comparable,
        "pct_comparable_datapoints": (n_comparable / n_datapoints * 100 if n_datapoints else 0.0),
        "measurements_in_comparable": int(assay_values[multi_assay].map(len).sum()),
        "represented_assays": n_assays,
        "assays_with_overlap": n_overlapping_assays,
        "pct_assays_with_overlap": (n_overlapping_assays / n_assays * 100 if n_assays else 0.0),
        "assays_without_overlap": n_assays - n_overlapping_assays,
    }
    if n_retrieved is not None:
        coverage["pct_measurements_in_comparable"] = (
            coverage["measurements_in_comparable"] / n_retrieved * 100 if n_retrieved else 0.0
        )
    return coverage


def format_cross_assay_coverage(
    coverage: dict[str, Union[int, float]],
    title: str = "CROSS-ASSAY COVERAGE",
    indent: str = "  ",
) -> str:
    """Render :func:`summarize_cross_assay_coverage` output as an aligned text block.

    Args:
        coverage: Output of :func:`summarize_cross_assay_coverage`.
        title: Heading placed above the block.
        indent: Prefix for the heading; entries are indented one level deeper.

    Returns:
        Multi-line string, without a trailing newline.
    """
    row_indent = indent * 2
    width = 36
    lines = [
        f"{indent}{title}",
        f"{row_indent}{'Aggregated datapoints:':<{width}s} {coverage['aggregated_datapoints']:>8,}",
        f"{row_indent}{DATAPOINT_OVERLAP_LABEL + ':':<{width}s} {coverage['comparable_datapoints']:>8,}"
        f"  ({coverage['pct_comparable_datapoints']:5.1f}%)",
    ]
    line = f"{row_indent}{'Measurements they pool:':<{width}s} {coverage['measurements_in_comparable']:>8,}"
    if "pct_measurements_in_comparable" in coverage:
        line += f"  ({coverage['pct_measurements_in_comparable']:5.1f}% of retrieved)"
    lines.append(line)
    lines.extend(
        [
            f"{row_indent}{'Assay IDs represented:':<{width}s} {coverage['represented_assays']:>8,}",
            f"{row_indent}{ASSAY_OVERLAP_LABEL + ':':<{width}s} {coverage['assays_with_overlap']:>8,}"
            f"  ({coverage['pct_assays_with_overlap']:5.1f}%)",
            f"{row_indent}{NO_ASSAY_OVERLAP_LABEL + ':':<{width}s} {coverage['assays_without_overlap']:>8,}"
            f"  ({100 - coverage['pct_assays_with_overlap']:5.1f}%)",
        ]
    )
    return "\n".join(lines)


def summarize_curation(
    name: str,
    retrieved: Optional[pd.DataFrame] = None,
    aggregated: Optional[pd.DataFrame] = None,
    comparisons: Optional[pd.DataFrame] = None,
    drop_flags: Optional[list[str]] = None,
    comment_column: str = DATA_DROPPING_COMMENT,
    comparison_comment_column: str = "dropping_comment",
    assay_column: str = ASSAY_ID,
    sep_str: str = "|",
) -> pd.DataFrame:
    """Account for one dataset at every level it is counted at, in one table.

    The same measurement is counted differently depending on the level: as a retrieved
    ChEMBL activity, as an aggregated compound-target datapoint pooling several
    activities, and as a pairwise cross-assay comparison. A percentage is meaningless
    without saying which of these is its denominator, so each level is reported
    separately and labelled.

    Args:
        name: Name of the dataset, repeated in the ``dataset`` column so that several
            datasets can be concatenated into one table.
        retrieved: Non-aggregated DataFrame, one row per retrieved measurement.
        aggregated: Aggregated DataFrame, one row per compound-target readout. Counted
            per row, and used for the cross-assay coverage row.
        comparisons: DataFrame of pairwise comparisons from
            :func:`~Capricho.analysis.explode_assay_comparability`, counted per pair.
        drop_flags: Flags being dropped. Their union is reported as the data omitted.
            If None, every flag found is reported.
        comment_column: Flag column of ``retrieved`` and ``aggregated``.
        comparison_comment_column: Flag column of ``comparisons``.
        assay_column: Column of ``aggregated`` holding the pooled assay identifiers.
        sep_str: Separator delimiting pooled measurements.

    Returns:
        Long-form DataFrame with columns ``dataset``, ``level``, ``flag``, ``kind``,
        ``n`` and ``pct``. ``kind`` is ``"flag"`` for a single reason, ``"total"`` for
        the union and the denominator, and ``"coverage"`` for the complementary shares
        of aggregated datapoints and represented assay identifiers with cross-assay overlap.
    """
    levels = [
        (RETRIEVED_LEVEL, retrieved, comment_column),
        (AGGREGATED_LEVEL, aggregated, comment_column),
        (COMPARISON_LEVEL, comparisons, comparison_comment_column),
    ]
    summaries = [
        summarize_flags(data, flags=drop_flags, comment_column=column, sep_str=sep_str).assign(
            dataset=name, level=level
        )
        for level, data, column in levels
        if data is not None
    ]
    if not summaries:
        raise ValueError("Pass at least one of `retrieved`, `aggregated` or `comparisons`.")

    if aggregated is not None and assay_column in aggregated.columns:
        coverage = summarize_cross_assay_coverage(aggregated, assay_column=assay_column, sep_str=sep_str)
        summaries.append(
            pd.DataFrame(
                {
                    "dataset": [name],
                    "level": [AGGREGATED_LEVEL],
                    "flag": [DATAPOINT_OVERLAP_LABEL],
                    "kind": ["coverage"],
                    "n": [coverage["comparable_datapoints"]],
                    "pct": [coverage["pct_comparable_datapoints"]],
                }
            )
        )
        summaries.append(
            pd.DataFrame(
                {
                    "flag": [ASSAY_OVERLAP_LABEL, NO_ASSAY_OVERLAP_LABEL, TOTAL_LABEL],
                    "kind": ["coverage", "coverage", "total"],
                    "n": [
                        coverage["assays_with_overlap"],
                        coverage["assays_without_overlap"],
                        coverage["represented_assays"],
                    ],
                    "pct": [
                        coverage["pct_assays_with_overlap"],
                        100 - coverage["pct_assays_with_overlap"],
                        100.0 if coverage["represented_assays"] else 0.0,
                    ],
                }
            ).assign(dataset=name, level=ASSAY_LEVEL)
        )

    return pd.concat(summaries, ignore_index=True)[["dataset", "level", "flag", "kind", "n", "pct"]]


def summarize_curation_by_group(
    group_column: str,
    retrieved: Optional[pd.DataFrame] = None,
    aggregated: Optional[pd.DataFrame] = None,
    comparisons: Optional[pd.DataFrame] = None,
    drop_flags: Optional[list[str]] = None,
    labels: Optional[dict] = None,
    comment_column: str = DATA_DROPPING_COMMENT,
    comparison_comment_column: str = "dropping_comment",
    assay_column: str = ASSAY_ID,
    sep_str: str = "|",
) -> pd.DataFrame:
    """Account for one dataset per group of ``group_column``, in one table.

    A dataset covering several targets or endpoints hides how unevenly they are curated:
    an aggregate coverage figure can look healthy while individual targets have almost no
    cross-assay overlap. This splits every frame on the same column and reports each group
    as its own dataset, so the groups can be compared side by side.

    Args:
        group_column: Column splitting the data into groups. Must be present in every
            frame passed, and single-valued in all of them.
        retrieved: Non-aggregated DataFrame, one row per retrieved measurement.
        aggregated: Aggregated DataFrame, one row per compound-target readout.
        comparisons: DataFrame of pairwise comparisons.
        drop_flags: Flags being dropped, applied identically to every group.
        labels: Maps a group value to the name reported for it. Values without an entry
            are reported under the group value itself.
        comment_column: Flag column of ``retrieved`` and ``aggregated``.
        comparison_comment_column: Flag column of ``comparisons``.
        assay_column: Column of ``aggregated`` holding the pooled assay identifiers.
        sep_str: Separator delimiting pooled measurements.

    Returns:
        The :func:`summarize_curation` table of every group, concatenated. Groups appear
        in the order they first occur in ``aggregated``, or in ``comparisons`` or
        ``retrieved`` when ``aggregated`` is not given.

    Raises:
        ValueError: If no frame is passed, or if a frame that is passed lacks
            ``group_column``.
    """
    given = {
        name: frame
        for name, frame in (
            ("retrieved", retrieved),
            ("aggregated", aggregated),
            ("comparisons", comparisons),
        )
        if frame is not None
    }
    if not given:
        raise ValueError("Pass at least one of `retrieved`, `aggregated` or `comparisons`.")

    missing = [name for name, frame in given.items() if group_column not in frame.columns]
    if missing:
        raise ValueError(
            f"Column '{group_column}' not found in {', '.join(missing)}. Every frame passed "
            "must carry it, otherwise a group would be counted against a denominator taken "
            "from the whole dataset."
        )

    ordering = next(given[name] for name in ("aggregated", "comparisons", "retrieved") if name in given)
    values = list(dict.fromkeys(ordering[group_column]))
    labels = labels or {}

    return pd.concat(
        [
            summarize_curation(
                labels.get(value, value),
                **{name: frame[frame[group_column] == value] for name, frame in given.items()},
                drop_flags=drop_flags,
                comment_column=comment_column,
                comparison_comment_column=comparison_comment_column,
                assay_column=assay_column,
                sep_str=sep_str,
            )
            for value in values
        ],
        ignore_index=True,
    )


#: Columns of :func:`coverage_table`, in order, and how they are spelled for a reader.
COVERAGE_COLUMN_LABELS = {
    "comparable_datapoints": "Datapoints measured in >1 assay",
    "aggregated_datapoints": "All aggregated datapoints",
    "datapoint_overlap_pct": "Datapoint overlap (%)",
    "assays_with_overlap": "Assays with overlap",
    "represented_assays": "Assays represented",
    "assay_overlap_pct": "Assay overlap (%)",
}

#: Coverage rows of :func:`summarize_curation`, and the wide columns they become.
_COVERAGE_SELECTIONS = [
    (
        AGGREGATED_LEVEL,
        DATAPOINT_OVERLAP_LABEL,
        {"n": "comparable_datapoints", "pct": "datapoint_overlap_pct"},
    ),
    (AGGREGATED_LEVEL, TOTAL_LABEL, {"n": "aggregated_datapoints"}),
    (ASSAY_LEVEL, ASSAY_OVERLAP_LABEL, {"n": "assays_with_overlap", "pct": "assay_overlap_pct"}),
    (ASSAY_LEVEL, TOTAL_LABEL, {"n": "represented_assays"}),
]


def coverage_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Reshape the cross-assay coverage rows of :func:`summarize_curation` into one row per dataset.

    The long form keeps every level in one column so that no percentage is quoted without
    its denominator. Reading coverage across datasets instead means comparing the same two
    shares side by side, which is what this returns: how much of the aggregated data is
    measured in more than one assay, and how many of the assays represented share a
    datapoint with another assay.

    Args:
        summary: Output of :func:`summarize_curation` or
            :func:`summarize_curation_by_group`, for one or several datasets.

    Returns:
        DataFrame with one row per dataset, in the order the datasets appear in ``summary``,
        and the columns named by :data:`COVERAGE_COLUMN_LABELS`, keyed by ``dataset``.

    Raises:
        ValueError: If ``summary`` carries no cross-assay coverage rows, which is the case
            when :func:`summarize_curation` ran without an ``aggregated`` frame or that
            frame had no assay identifier column.
    """
    table = None
    for level, flag, renames in _COVERAGE_SELECTIONS:
        rows = summary[(summary["level"] == level) & (summary["flag"] == flag)]
        if rows.empty:
            raise ValueError(
                f"No '{flag}' row at level '{level}'. Cross-assay coverage is only reported "
                "when `summarize_curation` is given an `aggregated` frame holding the pooled "
                "assay identifiers."
            )
        part = rows[["dataset", *renames]].rename(columns=renames)
        table = part if table is None else table.merge(part, on="dataset", validate="one_to_one")

    return table[["dataset", *COVERAGE_COLUMN_LABELS]]


def format_coverage_table(coverage: pd.DataFrame, dataset_label: str = "Dataset"):
    """Render :func:`coverage_table` output for display, with readable headers.

    Args:
        coverage: Output of :func:`coverage_table`.
        dataset_label: Header for the ``dataset`` column, naming what the rows are —
            for example ``"Endpoint"`` or ``"Isoform"``.

    Returns:
        A pandas ``Styler``: counts with thousands separators, shares to one decimal,
        and no index.
    """
    shown = coverage[["dataset", *COVERAGE_COLUMN_LABELS]].rename(
        columns={"dataset": dataset_label, **COVERAGE_COLUMN_LABELS}
    )
    return (
        shown.style.hide(axis="index")
        .format({label: "{:.1f}" for label in COVERAGE_COLUMN_LABELS.values() if "%" in label})
        .format({label: "{:,}" for label in COVERAGE_COLUMN_LABELS.values() if "%" not in label})
    )


def format_curation_summary(summary: pd.DataFrame, indent: str = "  ") -> str:
    """Render :func:`summarize_curation` output as one text block per dataset and level.

    Args:
        summary: Output of :func:`summarize_curation`, possibly for several datasets.
        indent: Prefix for the level headings.

    Returns:
        Multi-line string, without a trailing newline.
    """
    blocks = []
    for dataset, per_dataset in summary.groupby("dataset", sort=False):
        blocks.append(str(dataset))
        for level, per_level in per_dataset.groupby("level", sort=False):
            n_total = per_level.loc[per_level["flag"] == TOTAL_LABEL, "n"]
            denominator = f" — share of {n_total.iloc[0]:,}" if len(n_total) else ""
            non_coverage = per_level[per_level["kind"] != "coverage"]
            if (non_coverage["flag"] != TOTAL_LABEL).any():
                blocks.append(
                    format_flag_summary(
                        non_coverage,
                        title=f"{level}{denominator}",
                        indent=indent,
                    )
                )
            else:
                blocks.append(f"{indent}{level}{denominator}")
            width = _label_width(per_level)
            for row in per_level[per_level["kind"] == "coverage"].itertuples():
                blocks.append(f"{indent * 2}{row.flag + ':':<{width}s} {row.n:>8,}  ({row.pct:5.1f}%)")
    return "\n".join(blocks)
