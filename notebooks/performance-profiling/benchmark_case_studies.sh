#!/bin/bash
# Benchmark CAPRICHO case study commands for manuscript timing claims.
# Runs each command 3 times and reports mean/std wall-clock time and peak memory.
# Usage: ./benchmark_case_studies.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAPRICHO=(micromamba run -n capricho capricho)
MEASURE=(python3 "$SCRIPT_DIR/measure_command.py")
OUTDIR="$SCRIPT_DIR/output"
RESULTS_FILE="$SCRIPT_DIR/benchmark_results.csv"
N_RUNS=3

mkdir -p "$OUTDIR"

echo "run,command,wall_seconds,peak_rss_mb" > "$RESULTS_FILE"
echo "============================================"
echo "CAPRICHO Performance Benchmark"
echo "Runs per command: $N_RUNS"
echo "Output directory: $OUTDIR"
echo "Results file: $RESULTS_FILE"
echo "Date: $(date)"
echo "Machine: $(uname -m), $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB RAM", $1/1073741824}' 2>/dev/null || echo 'unknown')"
echo "============================================"
echo ""

run_benchmark() {
    local name="$1"
    shift
    local run_num="$1"
    shift

    echo "--- $name (run $run_num/$N_RUNS) ---"

    local output
    output=$("${MEASURE[@]}" "$@" 2>&1)

    # Print last 5 lines of command output (excludes BENCHMARK_RESULT line)
    echo "$output" | grep -v "^BENCHMARK_RESULT:" | tail -5

    # Parse the measurement line
    local result_line
    result_line=$(echo "$output" | grep "^BENCHMARK_RESULT:")
    local elapsed=$(echo "$result_line" | awk '{print $2}')
    local peak_rss=$(echo "$result_line" | awk '{print $3}')

    echo "$run_num,$name,$elapsed,$peak_rss" >> "$RESULTS_FILE"
    echo "  -> ${elapsed}s, peak RSS: ${peak_rss} MB"
    echo ""
}

# ============================================================
# Case Study 1: Ki dataset (26 targets)
# Target IDs extracted from rinikerlab/overlapping_assays IC50_datasets.yaml
# ============================================================
for i in $(seq 1 $N_RUNS); do
    run_benchmark "case1_ki" "$i" \
        "${CAPRICHO[@]}" get \
        --target-ids CHEMBL205,CHEMBL214,CHEMBL217,CHEMBL218,CHEMBL224,CHEMBL226,CHEMBL228,CHEMBL233,CHEMBL234,CHEMBL237,CHEMBL243,CHEMBL244,CHEMBL251,CHEMBL253,CHEMBL255,CHEMBL256,CHEMBL259,CHEMBL261,CHEMBL264,CHEMBL313,CHEMBL3155,CHEMBL3242,CHEMBL3371,CHEMBL344,CHEMBL3594,CHEMBL4153 \
        --chirality \
        --drop-unassigned-chiral \
        --require-doc-date \
        --curate-annotation-errors \
        --max-assay-size 100 \
        --min-assay-size 20 \
        --min-assay-overlap 5 \
        --strict-mutant-removal \
        --calculate-pchembl \
        --confidence-scores 9 \
        --id-columns assay_type,assay_cell_type \
        --bioactivity-type Ki \
        --output-path "$OUTDIR/curated-Ki.csv"
done

# ============================================================
# Case Study 1: IC50 dataset (39 targets)
# ============================================================
# First, resolve the IC50 target list
echo "Resolving IC50 target list from overlapping_assays repository..."
IC50_TARGETS=$(python3 -c "
import re
import numpy as np
from yaml import safe_load
from urllib.request import urlopen

url = 'https://raw.githubusercontent.com/rinikerlab/overlapping_assays/refs/heads/main/datasets/IC50_datasets.yaml'
datasets = safe_load(urlopen(url).read())
targets = []
for key in datasets['sources'].keys():
    targets.append(datasets['sources'][key]['description'])
targets = np.unique(targets).tolist()
targets = [re.compile(r'(CHEMBL\d+)').search(t).group(1) for t in targets]
print(','.join(targets))
")
echo "IC50 targets ($( echo "$IC50_TARGETS" | tr ',' '\n' | wc -l | tr -d ' ')): ${IC50_TARGETS:0:80}..."
echo ""

for i in $(seq 1 $N_RUNS); do
    run_benchmark "case1_ic50" "$i" \
        "${CAPRICHO[@]}" get \
        --target-ids "$IC50_TARGETS" \
        --chirality \
        --drop-unassigned-chiral \
        --require-doc-date \
        --curate-annotation-errors \
        --max-assay-size 100 \
        --min-assay-size 20 \
        --min-assay-overlap 5 \
        --strict-mutant-removal \
        --calculate-pchembl \
        --confidence-scores 9 \
        --id-columns assay_type,assay_cell_type \
        --bioactivity-type IC50 \
        --output-path "$OUTDIR/curated-IC50.csv"
done

# ============================================================
# Case Study 2: CYP dataset (14 targets)
# ============================================================
for i in $(seq 1 $N_RUNS); do
    run_benchmark "case2_cyp" "$i" \
        "${CAPRICHO[@]}" get \
        --target-ids CHEMBL340,CHEMBL289,CHEMBL3397,CHEMBL3622,CHEMBL3356,CHEMBL2722,CHEMBL1908,CHEMBL3379,CHEMBL3978,CHEMBL4878,CHEMBL2231,CHEMBL3721,CHEMBL5282,CHEMBL4729 \
        --drop-unassigned-chiral \
        --chirality \
        --min-assay-size 5 \
        --max-assay-size 100 \
        --assay-types A,B,F \
        --id-columns standard_type \
        --output-path "$OUTDIR/cyp_chembl_data.csv"
done

# ============================================================
# Case Study 3: Caco-2 A→B (92 assays)
# ============================================================
for i in $(seq 1 $N_RUNS); do
    run_benchmark "case3_caco2_a2b" "$i" \
        "${CAPRICHO[@]}" get \
        --assay-ids CHEMBL3529272,CHEMBL3430218,CHEMBL3529276,CHEMBL3788414,CHEMBL4618331,CHEMBL967838,CHEMBL1010166,CHEMBL2399272,CHEMBL5535226,CHEMBL3102303,CHEMBL3413801,CHEMBL5226219,CHEMBL5367468,CHEMBL1936906,CHEMBL4626019,CHEMBL2061608,CHEMBL4718821,CHEMBL4822451,CHEMBL2404820,CHEMBL3791483,CHEMBL1781718,CHEMBL2182708,CHEMBL3529287,CHEMBL1777560,CHEMBL1810827,CHEMBL1954393,CHEMBL3135115,CHEMBL3224753,CHEMBL3529284,CHEMBL4016544,CHEMBL898578,CHEMBL1780678,CHEMBL4353732,CHEMBL1953550,CHEMBL2026558,CHEMBL2186064,CHEMBL2321450,CHEMBL4775534,CHEMBL3776213,CHEMBL4144179,CHEMBL930649,CHEMBL1059634,CHEMBL3396820,CHEMBL4616467,CHEMBL4842267,CHEMBL5243021,CHEMBL1119778,CHEMBL1657599,CHEMBL2382568,CHEMBL3411223,CHEMBL4266470,CHEMBL4322186,CHEMBL1775251,CHEMBL2175485,CHEMBL3750318,CHEMBL3294034,CHEMBL3295547,CHEMBL4481174,CHEMBL4672670,CHEMBL967979,CHEMBL2424026,CHEMBL3124084,CHEMBL3620852,CHEMBL4143981,CHEMBL4157520,CHEMBL4414795,CHEMBL5615768,CHEMBL3368176,CHEMBL3371469,CHEMBL3829697,CHEMBL4351258,CHEMBL4672721,CHEMBL5112808,CHEMBL5166088,CHEMBL5340432,CHEMBL5349017,CHEMBL5384114,CHEMBL923278,CHEMBL935710,CHEMBL1780502,CHEMBL1780839,CHEMBL2399058,CHEMBL2400832,CHEMBL3300842,CHEMBL3779198,CHEMBL3860793,CHEMBL4731880,CHEMBL5217583,CHEMBL5340614,CHEMBL5638895,CHEMBL921232,CHEMBL923495 \
        --assay-types A \
        --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
        --value-column standard_value \
        --id-columns standard_units,assay_cell_type \
        --convert-units \
        --drop-unassigned-chiral \
        --output-path "$OUTDIR/caco2_a_to_b.csv"
done

# ============================================================
# Case Study 3: Caco-2 B→A (49 assays)
# ============================================================
for i in $(seq 1 $N_RUNS); do
    run_benchmark "case3_caco2_b2a" "$i" \
        "${CAPRICHO[@]}" get \
        --assay-ids CHEMBL3529273,CHEMBL3529277,CHEMBL3413802,CHEMBL1936907,CHEMBL2061609,CHEMBL4718841,CHEMBL3529288,CHEMBL3529285,CHEMBL2026704,CHEMBL2321447,CHEMBL3396835,CHEMBL1657600,CHEMBL4322187,CHEMBL3750319,CHEMBL3102302,CHEMBL3294035,CHEMBL4157521,CHEMBL5349018,CHEMBL5384115,CHEMBL923279,CHEMBL1780503,CHEMBL2400831,CHEMBL4731881,CHEMBL5388553,CHEMBL5638896,CHEMBL921231,CHEMBL923494,CHEMBL1100523,CHEMBL1959677,CHEMBL3368149,CHEMBL4477890,CHEMBL5141815,CHEMBL5503593,CHEMBL1060844,CHEMBL1943181,CHEMBL2026005,CHEMBL4037922,CHEMBL944381,CHEMBL1803338,CHEMBL2148941,CHEMBL2350101,CHEMBL3419538,CHEMBL4357336,CHEMBL4373230,CHEMBL4380529,CHEMBL5111948,CHEMBL5128826,CHEMBL5153793,CHEMBL967839 \
        --assay-types A \
        --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
        --value-column standard_value \
        --id-columns standard_units,assay_cell_type \
        --convert-units \
        --drop-unassigned-chiral \
        --output-path "$OUTDIR/caco2_b_to_a.csv"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "Benchmark Complete"
echo "============================================"
echo ""

python3 -c "
import csv
from collections import defaultdict
import statistics

times = defaultdict(list)
memory = defaultdict(list)
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        times[row['command']].append(float(row['wall_seconds']))
        memory[row['command']].append(float(row['peak_rss_mb']))

header = '{:<20} {:>10} {:>10} {:>10} {:>12} {:>12} {:>6}'.format(
    'Command', 'Mean (s)', 'Std (s)', 'Mean (min)', 'Peak RSS MB', 'Std RSS MB', 'Runs')
print(header)
print('-' * len(header))
for cmd in ['case1_ki', 'case1_ic50', 'case2_cyp', 'case3_caco2_a2b', 'case3_caco2_b2a']:
    if cmd in times:
        t = times[cmd]
        m = memory[cmd]
        t_mean = statistics.mean(t)
        t_std = statistics.stdev(t) if len(t) > 1 else 0
        m_mean = statistics.mean(m)
        m_std = statistics.stdev(m) if len(m) > 1 else 0
        print(f'{cmd:<20} {t_mean:>10.1f} {t_std:>10.1f} {t_mean/60:>10.1f} {m_mean:>12.0f} {m_std:>12.0f} {len(t):>6}')
"
