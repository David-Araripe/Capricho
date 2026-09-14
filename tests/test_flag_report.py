"""Tests for the quality-flag accounting reported by the CLI and the case studies."""

import pandas as pd
import pytest

from Capricho.analysis import DroppingComment
from Capricho.flag_report import (
    AGGREGATED_LEVEL,
    ANY_FLAG_LABEL,
    ANY_SELECTED_LABEL,
    ASSAY_LEVEL,
    ASSAY_OVERLAP_LABEL,
    COMPARISON_LEVEL,
    COVERAGE_COLUMN_LABELS,
    NO_ASSAY_OVERLAP_LABEL,
    RETAINED_LABEL,
    RETRIEVED_LEVEL,
    TOTAL_LABEL,
    UNFLAGGED_LABEL,
    coverage_table,
    flags_in_comment,
    format_coverage_table,
    format_cross_assay_coverage,
    format_curation_summary,
    format_flag_summary,
    measurement_comments,
    summarize_cross_assay_coverage,
    summarize_curation,
    summarize_curation_by_group,
    summarize_flags,
)


def count_of(summary: pd.DataFrame, flag: str) -> int:
    """Absolute count reported for a flag."""
    return int(summary.loc[summary["flag"] == flag, "n"].iloc[0])


def pct_of(summary: pd.DataFrame, flag: str) -> float:
    """Percentage reported for a flag."""
    return float(summary.loc[summary["flag"] == flag, "pct"].iloc[0])


class TestMeasurementComments:
    def test_splits_pooled_measurements(self):
        assert measurement_comments("A|B|C") == ["A", "B", "C"]

    def test_single_comment_stays_whole(self):
        assert measurement_comments("Undefined Stereochemistry") == ["Undefined Stereochemistry"]

    def test_missing_values_yield_one_empty_comment(self):
        assert measurement_comments(None) == [""]
        assert measurement_comments(float("nan")) == [""]
        assert measurement_comments("nan") == [""]

    def test_empty_positions_are_kept_so_measurements_stay_aligned(self):
        assert measurement_comments("|Potential Duplicate|") == ["", "Potential Duplicate", ""]


class TestFlagsInComment:
    def test_splits_flags_joined_on_one_measurement(self):
        assert flags_in_comment("Potential Duplicate & Undefined Stereochemistry") == {
            "Potential Duplicate",
            "Undefined Stereochemistry",
        }

    def test_normalizes_run_specific_thresholds(self):
        assert flags_in_comment("Assay size < 5") == {"Assay size <"}
        assert flags_in_comment("Assay size < 20") == {"Assay size <"}

    def test_empty_comment_has_no_flags(self):
        assert flags_in_comment("") == set()
        assert flags_in_comment("nan") == set()


class TestSummarizeFlags:
    @pytest.fixture
    def flagged(self):
        """Four measurements: two share a flag, one carries two flags, one is clean."""
        return pd.DataFrame(
            {
                "data_dropping_comment": [
                    "Undefined Stereochemistry",
                    "Undefined Stereochemistry & Assay size < 5",
                    "Potential Duplicate",
                    "",
                ]
            }
        )

    def test_counts_each_flag_and_its_share(self, flagged):
        summary = summarize_flags(flagged)

        assert count_of(summary, "Undefined Stereochemistry") == 2
        assert pct_of(summary, "Undefined Stereochemistry") == 50.0
        assert count_of(summary, "Assay size <") == 1
        assert count_of(summary, "Potential Duplicate") == 1

    def test_union_counts_co_flagged_entries_once(self, flagged):
        summary = summarize_flags(flagged)

        # 2 + 1 + 1 = 4 per-flag hits, but only 3 distinct flagged measurements.
        assert summary.loc[summary["kind"] == "flag", "n"].sum() == 4
        assert count_of(summary, ANY_FLAG_LABEL) == 3
        assert pct_of(summary, ANY_FLAG_LABEL) == 75.0

    def test_totals_partition_the_dataset(self, flagged):
        summary = summarize_flags(flagged)

        assert count_of(summary, UNFLAGGED_LABEL) == 1
        assert count_of(summary, TOTAL_LABEL) == len(flagged)
        assert count_of(summary, ANY_FLAG_LABEL) + count_of(summary, UNFLAGGED_LABEL) == len(flagged)

    def test_complement_is_named_for_a_filter_only_when_flags_are_selected(self, flagged):
        """Without a selection the summary describes the data; with one it describes a filter."""
        assert UNFLAGGED_LABEL in set(summarize_flags(flagged)["flag"])
        assert RETAINED_LABEL in set(summarize_flags(flagged, flags=["Potential Duplicate"])["flag"])

    def test_selected_flags_report_only_the_omitted_data(self, flagged):
        summary = summarize_flags(flagged, flags=[DroppingComment.UNDEFINED_STEREOCHEMISTRY.value])

        assert list(summary.loc[summary["kind"] == "flag", "flag"]) == ["Undefined Stereochemistry"]
        assert count_of(summary, ANY_SELECTED_LABEL) == 2
        assert count_of(summary, RETAINED_LABEL) == 2

    def test_selected_flag_absent_from_data_is_reported_as_zero(self, flagged):
        summary = summarize_flags(flagged, flags=[DroppingComment.MIXTURE_IN_SMILES.value])

        assert count_of(summary, "Mixture in SMILES") == 0
        assert count_of(summary, ANY_SELECTED_LABEL) == 0
        assert count_of(summary, RETAINED_LABEL) == len(flagged)

    def test_thresholds_of_the_same_flag_are_pooled(self):
        data = pd.DataFrame({"data_dropping_comment": ["Assay size < 5", "Assay size < 20"]})

        summary = summarize_flags(data)

        assert count_of(summary, "Assay size <") == 2

    def test_selected_flags_are_normalized_before_matching(self):
        data = pd.DataFrame({"data_dropping_comment": ["Assay size < 5"]})

        summary = summarize_flags(data, flags=["Assay size < 5"])

        assert count_of(summary, "Assay size <") == 1

    def test_overlapping_flag_names_are_not_double_counted(self):
        """ "Insufficient assay overlap" is a prefix of the metadata-matching variant."""
        data = pd.DataFrame(
            {
                "data_dropping_comment": [
                    "Insufficient assay overlap (min_overlap=5)",
                    "Insufficient assay overlap with metadata matching (min_overlap=5)",
                ]
            }
        )

        summary = summarize_flags(data)

        assert count_of(summary, "Insufficient assay overlap") == 1
        assert count_of(summary, "Insufficient assay overlap with metadata matching") == 1
        assert count_of(summary, ANY_FLAG_LABEL) == 2

    def test_flags_are_ordered_by_descending_count(self):
        data = pd.DataFrame(
            {"data_dropping_comment": ["Potential Duplicate"] * 3 + ["Undefined Stereochemistry"]}
        )

        summary = summarize_flags(data)

        assert list(summary.loc[summary["kind"] == "flag", "flag"]) == [
            "Potential Duplicate",
            "Undefined Stereochemistry",
        ]

    def test_empty_dataset_reports_zeros_without_dividing_by_zero(self):
        summary = summarize_flags(pd.DataFrame({"data_dropping_comment": []}))

        assert count_of(summary, TOTAL_LABEL) == 0
        assert pct_of(summary, ANY_FLAG_LABEL) == 0.0

    def test_missing_column_is_an_error(self):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            summarize_flags(pd.DataFrame({"other": [1]}))

    def test_alternative_comment_column(self):
        """Pairwise comparisons carry their flags in `dropping_comment`."""
        data = pd.DataFrame({"dropping_comment": ["Undefined Stereochemistry|"]})

        summary = summarize_flags(data, comment_column="dropping_comment")

        assert count_of(summary, "Undefined Stereochemistry") == 1


class TestAggregatedCounting:
    @pytest.fixture
    def aggregated(self):
        """Two rows pooling five measurements, two of which are flagged."""
        return pd.DataFrame(
            {
                "data_dropping_comment": [
                    "Undefined Stereochemistry||Potential Duplicate",
                    "|",
                ]
            }
        )

    def test_row_counting_marks_a_row_carrying_any_flagged_measurement(self, aggregated):
        summary = summarize_flags(aggregated)

        assert count_of(summary, TOTAL_LABEL) == 2
        assert count_of(summary, ANY_FLAG_LABEL) == 1
        assert count_of(summary, "Undefined Stereochemistry") == 1

    def test_measurement_counting_uses_pooled_measurements_as_denominator(self, aggregated):
        summary = summarize_flags(aggregated, per_measurement=True)

        assert count_of(summary, TOTAL_LABEL) == 5
        assert count_of(summary, ANY_FLAG_LABEL) == 2
        assert count_of(summary, UNFLAGGED_LABEL) == 3
        assert pct_of(summary, ANY_FLAG_LABEL) == 40.0

    def test_row_and_measurement_counts_differ_on_aggregated_data(self, aggregated):
        by_row = summarize_flags(aggregated)
        by_measurement = summarize_flags(aggregated, per_measurement=True)

        assert count_of(by_row, TOTAL_LABEL) < count_of(by_measurement, TOTAL_LABEL)


class TestSummaryMatchesFiltering:
    """The reported omission must equal what dropping those flags actually removes."""

    def test_union_equals_rows_removed_by_filter_dropping_flags(self):
        from Capricho.core.pandas_helper import filter_dropping_flags

        data = pd.DataFrame(
            {
                "data_dropping_comment": [
                    "Undefined Stereochemistry",
                    "Undefined Stereochemistry & Potential Duplicate",
                    "Potential Duplicate",
                    "",
                    "Mixture in SMILES",
                ]
            }
        )
        flags = [
            DroppingComment.UNDEFINED_STEREOCHEMISTRY.value,
            DroppingComment.POTENTIAL_DUPLICATE.value,
        ]

        summary = summarize_flags(data, flags=flags)
        filtered = filter_dropping_flags(data, flags)

        assert count_of(summary, ANY_SELECTED_LABEL) == len(data) - len(filtered)
        assert count_of(summary, RETAINED_LABEL) == len(filtered)


class TestFormatFlagSummary:
    def test_reports_absolute_number_and_percentage(self):
        data = pd.DataFrame({"data_dropping_comment": ["Potential Duplicate", ""]})

        text = format_flag_summary(summarize_flags(data))

        assert "Potential Duplicate:" in text
        assert "1" in text and "50.0%" in text

    def test_totals_are_separated_from_the_per_flag_rows(self):
        data = pd.DataFrame({"data_dropping_comment": ["Potential Duplicate"]})

        lines = format_flag_summary(summarize_flags(data)).splitlines()

        assert lines[0].strip() == "QUALITY FLAGS"
        assert any(set(line.strip()) == {"-"} for line in lines)
        assert ANY_FLAG_LABEL in lines[-3]
        assert TOTAL_LABEL in lines[-1]

    def test_unflagged_dataset_says_so(self):
        data = pd.DataFrame({"data_dropping_comment": ["", ""]})

        text = format_flag_summary(summarize_flags(data))

        assert "(none)" in text


class TestCrossAssayCoverage:
    @pytest.fixture
    def aggregated(self):
        """Three compounds; only the first two were measured in more than one assay."""
        return pd.DataFrame({"assay_chembl_id": ["CHEMBL1|CHEMBL2|CHEMBL3", "CHEMBL1|CHEMBL2", "CHEMBL4"]})

    def test_counts_datapoints_reachable_by_cross_assay_comparison(self, aggregated):
        coverage = summarize_cross_assay_coverage(aggregated)

        assert coverage["aggregated_datapoints"] == 3
        assert coverage["comparable_datapoints"] == 2
        assert coverage["pct_comparable_datapoints"] == pytest.approx(66.67, abs=0.01)

    def test_counts_the_measurements_those_datapoints_pool(self, aggregated):
        coverage = summarize_cross_assay_coverage(aggregated)

        assert coverage["measurements_in_comparable"] == 5

    def test_counts_assays_with_and_without_cross_assay_overlap(self, aggregated):
        coverage = summarize_cross_assay_coverage(aggregated)

        assert coverage["represented_assays"] == 4
        assert coverage["assays_with_overlap"] == 3
        assert coverage["pct_assays_with_overlap"] == 75.0
        assert coverage["assays_without_overlap"] == 1

    def test_share_of_retrieved_measurements_when_known(self, aggregated):
        coverage = summarize_cross_assay_coverage(aggregated, n_retrieved=10)

        assert coverage["pct_measurements_in_comparable"] == 50.0

    def test_share_of_retrieved_is_omitted_when_unknown(self, aggregated):
        assert "pct_measurements_in_comparable" not in summarize_cross_assay_coverage(aggregated)

    def test_single_assay_dataset_has_no_coverage(self):
        data = pd.DataFrame({"assay_chembl_id": ["CHEMBL1", "CHEMBL1|CHEMBL1"]})

        coverage = summarize_cross_assay_coverage(data)

        assert coverage["comparable_datapoints"] == 0
        assert coverage["pct_comparable_datapoints"] == 0.0
        assert coverage["represented_assays"] == 1
        assert coverage["assays_with_overlap"] == 0
        assert coverage["assays_without_overlap"] == 1

    def test_missing_column_is_an_error(self):
        with pytest.raises(ValueError, match="not found in DataFrame"):
            summarize_cross_assay_coverage(pd.DataFrame({"other": [1]}))

    def test_formats_counts_and_shares(self, aggregated):
        text = format_cross_assay_coverage(summarize_cross_assay_coverage(aggregated, n_retrieved=10))

        assert "Measured in >1 assay:" in text
        assert "66.7%" in text
        assert "50.0% of retrieved" in text
        assert "Has cross-assay overlap:" in text
        assert "75.0%" in text
        assert "No cross-assay overlap:" in text


class TestSummarizeCuration:
    @pytest.fixture
    def retrieved(self):
        """Four retrieved measurements, one of them flagged."""
        return pd.DataFrame({"data_dropping_comment": ["Undefined Stereochemistry", "", "", ""]})

    @pytest.fixture
    def aggregated(self):
        """Three datapoints pooling the four measurements; only the first is comparable."""
        return pd.DataFrame(
            {
                "assay_chembl_id": ["CHEMBL1|CHEMBL2", "CHEMBL1", "CHEMBL3"],
                "data_dropping_comment": ["Undefined Stereochemistry|", "", ""],
            }
        )

    @pytest.fixture
    def comparisons(self):
        """The single pair the comparable datapoint contributes."""
        return pd.DataFrame({"dropping_comment": ["Undefined Stereochemistry"]})

    def test_each_level_keeps_its_own_denominator(self, retrieved, aggregated, comparisons):
        summary = summarize_curation(
            "Caco-2 A2B", retrieved=retrieved, aggregated=aggregated, comparisons=comparisons
        )

        totals = summary[summary["flag"] == TOTAL_LABEL].set_index("level")["n"]
        assert totals[RETRIEVED_LEVEL] == 4
        assert totals[AGGREGATED_LEVEL] == 3
        assert totals[COMPARISON_LEVEL] == 1
        assert totals[ASSAY_LEVEL] == 3

    def test_omission_is_reported_per_level(self, retrieved, aggregated, comparisons):
        summary = summarize_curation(
            "Caco-2 A2B",
            retrieved=retrieved,
            aggregated=aggregated,
            comparisons=comparisons,
            drop_flags=[DroppingComment.UNDEFINED_STEREOCHEMISTRY.value],
        )

        omitted = summary[summary["flag"] == ANY_SELECTED_LABEL].set_index("level")
        assert omitted.loc[RETRIEVED_LEVEL, "n"] == 1
        assert omitted.loc[RETRIEVED_LEVEL, "pct"] == 25.0
        assert omitted.loc[AGGREGATED_LEVEL, "n"] == 1
        assert omitted.loc[AGGREGATED_LEVEL, "pct"] == pytest.approx(33.33, abs=0.01)
        assert omitted.loc[COMPARISON_LEVEL, "pct"] == 100.0

    def test_reports_the_share_reachable_by_cross_assay_comparison(self, retrieved, aggregated):
        summary = summarize_curation("Caco-2 A2B", retrieved=retrieved, aggregated=aggregated)

        coverage = summary[(summary["kind"] == "coverage") & (summary["level"] == AGGREGATED_LEVEL)]
        assert coverage["n"].iloc[0] == 1
        assert coverage["pct"].iloc[0] == pytest.approx(33.33, abs=0.01)

    def test_reports_assays_with_and_without_overlap(self, aggregated):
        summary = summarize_curation("Caco-2 A2B", aggregated=aggregated)
        assay_coverage = summary[summary["level"] == ASSAY_LEVEL].set_index("flag")

        assert assay_coverage.loc[ASSAY_OVERLAP_LABEL, "n"] == 2
        assert assay_coverage.loc[ASSAY_OVERLAP_LABEL, "pct"] == pytest.approx(66.67, abs=0.01)
        assert assay_coverage.loc[NO_ASSAY_OVERLAP_LABEL, "n"] == 1
        assert assay_coverage.loc[NO_ASSAY_OVERLAP_LABEL, "pct"] == pytest.approx(33.33, abs=0.01)
        assert assay_coverage.loc[TOTAL_LABEL, "n"] == 3

    def test_dataset_name_allows_concatenating_several_datasets(self, retrieved):
        combined = pd.concat(
            [
                summarize_curation("A2B", retrieved=retrieved),
                summarize_curation("B2A", retrieved=retrieved),
            ]
        )

        assert set(combined["dataset"]) == {"A2B", "B2A"}

    def test_levels_left_out_are_absent(self, retrieved):
        summary = summarize_curation("A2B", retrieved=retrieved)

        assert set(summary["level"]) == {RETRIEVED_LEVEL}

    def test_no_level_given_is_an_error(self):
        with pytest.raises(ValueError, match="at least one"):
            summarize_curation("A2B")

    def test_formats_one_block_per_level(self, retrieved, aggregated, comparisons):
        text = format_curation_summary(
            summarize_curation(
                "Caco-2 A2B", retrieved=retrieved, aggregated=aggregated, comparisons=comparisons
            )
        )

        assert text.splitlines()[0] == "Caco-2 A2B"
        assert RETRIEVED_LEVEL in text
        assert AGGREGATED_LEVEL in text
        assert COMPARISON_LEVEL in text
        assert ASSAY_LEVEL in text
        assert "Measured in >1 assay:" in text
        assert ASSAY_OVERLAP_LABEL in text
        assert NO_ASSAY_OVERLAP_LABEL in text


class TestSummarizeCurationByGroup:
    """Tests for splitting a multi-target dataset into one reported dataset per group."""

    @pytest.fixture
    def retrieved(self):
        """Five measurements over two targets; the flagged one belongs to T1."""
        return pd.DataFrame(
            {
                "target_chembl_id": ["T1", "T1", "T1", "T2", "T2"],
                "data_dropping_comment": ["Undefined Stereochemistry", "", "", "", ""],
            }
        )

    @pytest.fixture
    def aggregated(self):
        """Four datapoints, two per target; one of each target's is comparable."""
        return pd.DataFrame(
            {
                "target_chembl_id": ["T1", "T1", "T2", "T2"],
                "assay_chembl_id": ["CHEMBL1|CHEMBL2", "CHEMBL1", "CHEMBL3|CHEMBL4", "CHEMBL5"],
                "data_dropping_comment": ["Undefined Stereochemistry|", "", "", ""],
            }
        )

    def test_reports_one_dataset_per_group(self, retrieved, aggregated):
        summary = summarize_curation_by_group("target_chembl_id", retrieved, aggregated)

        assert list(dict.fromkeys(summary["dataset"])) == ["T1", "T2"]

    def test_labels_rename_the_reported_groups(self, retrieved, aggregated):
        summary = summarize_curation_by_group(
            "target_chembl_id", retrieved, aggregated, labels={"T1": "CYP 3A4"}
        )

        # T2 has no entry, so it keeps the group value it was split on.
        assert list(dict.fromkeys(summary["dataset"])) == ["CYP 3A4", "T2"]

    def test_groups_partition_every_level(self, retrieved, aggregated):
        summary = summarize_curation_by_group("target_chembl_id", retrieved, aggregated)
        totals = summary[summary["flag"] == TOTAL_LABEL].set_index(["dataset", "level"])["n"]

        assert totals[("T1", RETRIEVED_LEVEL)] == 3
        assert totals[("T2", RETRIEVED_LEVEL)] == 2
        assert totals[("T1", AGGREGATED_LEVEL)] == 2
        assert totals[("T2", AGGREGATED_LEVEL)] == 2

    def test_coverage_is_computed_within_each_group(self, retrieved, aggregated):
        summary = summarize_curation_by_group("target_chembl_id", retrieved, aggregated)
        assays = summary[summary["level"] == ASSAY_LEVEL].set_index(["dataset", "flag"])["n"]

        # T2's CHEMBL5 is isolated, so only two of its three assays overlap.
        assert assays[("T1", TOTAL_LABEL)] == 2
        assert assays[("T1", ASSAY_OVERLAP_LABEL)] == 2
        assert assays[("T2", TOTAL_LABEL)] == 3
        assert assays[("T2", ASSAY_OVERLAP_LABEL)] == 2

    def test_flags_are_counted_against_the_group_denominator(self, retrieved, aggregated):
        summary = summarize_curation_by_group(
            "target_chembl_id",
            retrieved,
            aggregated,
            drop_flags=[DroppingComment.UNDEFINED_STEREOCHEMISTRY.value],
        )
        omitted = summary[
            (summary["flag"] == ANY_SELECTED_LABEL) & (summary["level"] == RETRIEVED_LEVEL)
        ].set_index("dataset")

        # One of T1's three measurements, not one of the five in the whole dataset.
        assert omitted.loc["T1", "n"] == 1
        assert omitted.loc["T1", "pct"] == pytest.approx(33.33, abs=0.01)
        assert omitted.loc["T2", "n"] == 0

    def test_frame_without_the_group_column_is_an_error(self, aggregated):
        retrieved = pd.DataFrame({"data_dropping_comment": [""]})

        with pytest.raises(ValueError, match="retrieved"):
            summarize_curation_by_group("target_chembl_id", retrieved, aggregated)

    def test_no_level_given_is_an_error(self):
        with pytest.raises(ValueError, match="at least one"):
            summarize_curation_by_group("target_chembl_id")


class TestCoverageTable:
    """Tests for reshaping the coverage rows into one row per dataset."""

    @pytest.fixture
    def aggregated(self):
        """Three datapoints; only the first is measured in more than one assay."""
        return pd.DataFrame(
            {
                "assay_chembl_id": ["CHEMBL1|CHEMBL2", "CHEMBL1", "CHEMBL3"],
                "data_dropping_comment": ["", "", ""],
            }
        )

    @pytest.fixture
    def summary(self, aggregated):
        """Two datasets, the second holding half of the first."""
        return pd.concat(
            [
                summarize_curation("A2B", aggregated=aggregated),
                summarize_curation("B2A", aggregated=aggregated.head(2)),
            ],
            ignore_index=True,
        )

    def test_reports_one_row_per_dataset_in_order(self, summary):
        assert list(coverage_table(summary)["dataset"]) == ["A2B", "B2A"]

    def test_datapoint_overlap_matches_the_long_form(self, summary, aggregated):
        table = coverage_table(summary).set_index("dataset")

        assert table.loc["A2B", "comparable_datapoints"] == 1
        assert table.loc["A2B", "aggregated_datapoints"] == 3
        assert table.loc["A2B", "datapoint_overlap_pct"] == pytest.approx(33.33, abs=0.01)

    def test_assay_overlap_keeps_its_own_denominator(self, summary):
        table = coverage_table(summary).set_index("dataset")

        # Two of the three assay identifiers overlap; the datapoint denominator is also 3,
        # so a wrong denominator would go unnoticed on A2B but not on B2A.
        assert table.loc["A2B", "assays_with_overlap"] == 2
        assert table.loc["A2B", "represented_assays"] == 3
        assert table.loc["B2A", "assays_with_overlap"] == 2
        assert table.loc["B2A", "represented_assays"] == 2
        assert table.loc["B2A", "assay_overlap_pct"] == 100.0

    def test_columns_are_the_documented_ones(self, summary):
        """The wide schema is the contract `plot_cross_assay_coverage` reads."""
        assert list(coverage_table(summary).columns) == ["dataset", *COVERAGE_COLUMN_LABELS]

    def test_display_headers_are_readable(self, summary):
        """Every column reaches the reader with a prose header and no index."""
        rendered = format_coverage_table(coverage_table(summary), dataset_label="Endpoint").to_html()

        assert "Endpoint" in rendered
        for label in COVERAGE_COLUMN_LABELS.values():
            assert label in rendered

    def test_summary_without_coverage_rows_is_an_error(self):
        summary = summarize_curation("A2B", comparisons=pd.DataFrame({"dropping_comment": [""]}))

        with pytest.raises(ValueError, match="Measured in >1 assay"):
            coverage_table(summary)
