# Key Concepts

Understanding these core concepts will help you use CAPRICHO effectively and make informed decisions about data curation.

## Compound Equality

One of the most important decisions in bioactivity data analysis is determining when two compound entries represent the same molecule.

### Connectivity-Based (Default)

The `connectivity` method identifies compounds by their molecular graph. It's based on the first 14 characters of the InChIKey, which encode atom connectivity but ignore stereochemistry and tautomerism:

```bash
capricho get --target-ids CHEMBL203 --compound-equality connectivity
```

Because the connectivity layer does not encode stereochemistry, stereoisomers (e.g., R/S enantiomers) are merged during aggregation. To reflect this, CAPRICHO strips stereochemistry from the output SMILES when using connectivity mode, preventing the output from misleadingly retaining an arbitrary enantiomer's representation.

**Advantages:**
- Robust to tautomers: InChI normalizes mobile-hydrogen tautomers (e.g., amide/imidic acid shifts between N, O, S atoms), so tautomeric forms are correctly identified as the same compound
- Deterministic and computationally lightweight

**Limitations:**
- Merges stereoisomers that may have different biological activity
- Does not normalize all tautomers

**Use When:**
- Stereochemistry information is unreliable or inconsistent across sources
- You want to combine data from different stereoisomers and tautomers into a single entry

### InChI-Based

The `inchi` method uses the complete standard InChI generated from each standardized SMILES:

```bash
capricho get --target-ids CHEMBL203 --compound-equality inchi
```

Unlike the connectivity layer alone, a full InChI preserves specified stereochemistry and other
InChI layers. The output includes an `inchi` column that can also be selected in downstream
commands such as `capricho prepare --compound-col inchi`.

**Use When:**
- Specified stereochemistry must remain part of compound identity
- You want a transparent, non-hashed standard identifier

### InChIKey-Based

The `inchikey` method uses the complete 27-character hash of the standard InChI:

```bash
capricho get --target-ids CHEMBL203 --compound-equality inchikey
```

It preserves the distinctions encoded by the full InChI—including specified stereochemistry—in
a compact identifier. The output includes an `inchikey` column. Because an InChIKey is a hash,
collisions are theoretically possible, although very unlikely for ordinary chemical datasets.
CAPRICHO retains `connectivity` in every aggregated output for backward compatibility and adds
`inchi` or `inchikey` when that full identifier is selected.

**Use When:**
- You need stereo-aware identity in a compact, database-friendly form
- You want to exchange identifiers with systems that use full InChIKeys

### Fingerprint-Based

The `mixed_fp` method uses a concatenation of ECFP4 (Morgan, radius 2) and RDKit path-based fingerprints to determine compound identity:

```bash
capricho get --target-ids CHEMBL203 --compound-equality mixed_fp
```

Using two fingerprint types covers some failure modes of each individual method — ECFP4 captures circular substructure environments while RDKit fingerprints capture path-based features. From those, only ECFP4 enables stereochemical distinctions and is used depending on the presence of the `--chirality` flag.

**Advantages:**
- Maintains stereochemical distinctions (with `--chirality`)

**Limitations:**
- Sensitive to tautomers: different tautomeric forms produce different fingerprint bit vectors, potentially treating the same compound as two different entries
- Susceptible to bit collisions in molecules with repetitive structural patterns (e.g., peptides, long lipid chains), where distinct substructures may hash to the same bits
- Computationally heavier than connectivity

**Use When:**
- Stereochemistry is important for your analysis
- Building focused datasets for SAR studies
- Working with data where tautomer variation is minimal

### SMILES-Based

The `smiles` method uses standardized SMILES strings directly for exact matching:

```bash
capricho get --target-ids CHEMBL203 --compound-equality smiles
```

**Advantages:**
- Simple and transparent matching logic
- No additional computation (InChI or fingerprints)

**Limitations:**
- Sensitive to tautomers: different tautomeric forms produce different SMILES strings
- Relies entirely on the standardization performed by the ChEMBL structure pipeline

## ChEMBL Backends

CAPRICHO supports two different ways to access ChEMBL data, each with distinct advantages.

### Local Database Backend (Default)

The `downloader` backend uses a local SQLite database:

```bash
capricho download  # One-time setup
capricho get --target-ids CHEMBL203 --chembl-backend downloader
```

**Advantages:**
- Much faster for large queries
- Works offline after initial download
- Consistent performance
- Full SQL query capabilities

**Requirements:**
- ~25GB disk space for full ChEMBL database
- Initial download time

**Best For:**
- Large-scale data mining
- Repeated queries
- Complex filtering requirements
- Offline analysis

### Web API Backend

The `webresource` backend queries the live ChEMBL web API:

```bash
capricho get --target-ids CHEMBL203 --chembl-backend webresource
```

**Advantages:**
- No local storage required
- Always uses latest data
- No setup time
- Good for small queries

**Limitations:**
- Slower for large queries
- Requires internet connection
- Subject to API rate limits

**Best For:**
- Small, targeted queries
- One-off analyses
- When disk space is limited

## Data Flagging

A core principle of CAPRICHO is **never silently dropping data**. Instead of removing problematic entries during the `get` command, CAPRICHO flags them and lets users decide what to filter during the `prepare` step.

### Two Types of Flags

CAPRICHO maintains two separate flag columns:

**`data_dropping_comment`** - Quality flags indicating data concerns:
- Potential duplicates across documents
- Data validity comments from ChEMBL
- Unit annotation errors (3.0 or 6.0 log unit differences)
- Patent sources
- Undefined stereochemistry
- Assay size issues
- Insufficient assay overlap

These include max curation standards from [Landrum & Riniker (2024)](https://doi.org/10.1021/acs.jcim.4c00049).

**`data_processing_comment`** - Processing flags documenting transformations:
- Salt/solvent removal from SMILES
- SMILES standardization
- pChEMBL calculation
- Unit conversions

### The Two-Phase Workflow

1. **`capricho get`**: Fetches and curates data, adding flags to all entries
2. **`capricho prepare`**: Filters data based on quality flags according to your project's needs

This separation ensures transparency - you can inspect flagged data before deciding what to remove:

```bash
# Step 1: Get data with all flags
capricho get --target-ids CHEMBL203 -o egfr_data.csv

# Step 2: Filter based on your quality requirements
capricho prepare -i egfr_data.csv -o egfr_clean.csv \
    --drop-potential-duplicate \
    --drop-data-validity
```

## Data Aggregation

CAPRICHO provides several options for handling duplicate measurements and aggregating data.

### Target Mutations

By default, CAPRICHO treats different target mutations as separate entities. Use `--aggregate-mutants` to combine them:

```bash
capricho get --target-ids CHEMBL203 --aggregate-mutants
```

This is useful when you want to study the target in general rather than specific mutations.

### Repeated downstream activity identifiers

Rows kept separate by mutation or `--id-columns` can still share the selected compound
identifier and `target_chembl_id`. CAPRICHO labels every member of such a group in the
`shared_identifier_group` output column. With the default method, the compound identifier is
`connectivity`; `inchi`, `inchikey`, and `smiles` are used when those equality methods are
selected. The label is a group identifier, not a quality
flag: the rows may represent valid, scientifically distinct readouts. Singleton rows contain
a missing value.

```python
# Each index is a group label and each value is that group's number of rows.
data.shared_identifier_group.value_counts()
```

Before pivoting or modeling on compound and target alone, inspect these groups. Preserve the
distinction by including the relevant fields in the downstream task identifier—for example,
`capricho prepare --id-columns mutation`—or deliberately filter or combine the group when
that is scientifically appropriate. CAPRICHO's INFO message reports the fields that vary and
shows all members of its sampled groups where the display limit permits.

### Metadata Columns

Include additional metadata in your analysis:

```bash
capricho get --target-ids CHEMBL203 --metadata-columns organism,tissue,cell_type
```

These columns are preserved during aggregation and can help you understand data heterogeneity.

(non-pchembl-aggregation)=
## Non-pChEMBL Aggregation

By default, CAPRICHO aggregates bioactivity data using the `pchembl_value` column, which represents -log10(molar) potency values. However, many ChEMBL assays (especially ADMET assays) report measurements that aren't suitable for pChEMBL conversion, such as:

- **Permeability** (e.g., Caco-2 apparent permeability in cm/s)
- **Percent inhibition** (e.g., % inhibition at a fixed concentration)
- **Clearance** (e.g., mL/min/kg)
- **Half-life** (e.g., hours)
- **Solubility** (e.g., µg/mL)

For these measurements, use `--value-column standard_value`:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279 \
  --assay-types A \
  --value-column standard_value \
  --output-path permeability_data.csv
```

### Key Differences from pChEMBL Aggregation

| Aspect | pchembl_value (default) | standard_value |
|--------|------------------------|----------------|
| **Mean type** | Geometric mean | Arithmetic mean |
| **Units** | Always molar (-log10) | Original units preserved |
| **Use case** | Potency measurements (IC50, Ki, etc.) | ADMET, physicochemical properties |

### Important: Preventing Unit Mixing

When aggregating on `standard_value`, ensure you don't inadvertently combine measurements with different units. Use `--id-columns` to group by unit type:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279 \
  --value-column standard_value \
  --id-columns standard_units \
  --output-path permeability_data.csv
```

This creates separate aggregations for measurements with different `standard_units` values.

(unit-conversion)=
## Unit Conversion

ChEMBL contains bioactivity data reported in many different units, even for the same measurement type. For example, permeability might be reported as `cm/s`, `nm/s`, or `10^-6 cm/s`. This heterogeneity makes cross-study aggregation challenging.

The `--convert-units` flag enables automatic unit conversion before aggregation:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279 \
  --value-column standard_value \
  --convert-units \
  --output-path permeability_data.csv
```

### Supported Unit Families

| Family | Target Unit | Source Units |
|--------|------------|--------------|
| **Permeability** | `10^-6 cm/s` | `cm/s`, `nm/s`, `ucm/s`, `10'-6 cm/s`, etc. |
| **Molar concentration** | `nM` | `uM`, `µM`, `mM`, `pM`, `M` |
| **Mass concentration** | `ug/mL` | `ng/ml`, `mg/ml`, `mg/L`, `pg/ml` |
| **Dose** | `mg/kg` | `ug/kg`, `ug.kg-1`, `mg.kg-1` |
| **Time** | `hr` | `min`, `s`, `ms`, `day` |

### Transparency

All unit conversions are logged and tracked in the `data_processing_comment` column. This ensures you can always trace which measurements were converted and by what factor.

### Example: Caco-2 Permeability Dataset

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
  --value-column standard_value \
  --convert-units \
  --id-columns standard_units,assay_cell_type \
  --drop-unassigned-chiral \
  --output-path caco2_permeability.csv
```

This command:
1. Fetches permeability data from specific Caco-2 assays
2. Converts all permeability units to `10^-6 cm/s`
3. Aggregates using arithmetic mean on `standard_value`
4. Groups by cell type to maintain biological context

## Confidence Scoring

ChEMBL assigns confidence scores (0-9) based on target assignment certainty. See the [ChEMBL documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions#what-is-the-confidence-score) for full details.

| Score | Description |
|-------|-------------|
| 9 | Direct single protein target assigned |
| 8 | Homologous single protein target assigned |
| 7 | Direct protein complex subunits assigned |
| 6 | Homologous protein complex subunits assigned |
| 5 | Multiple direct protein targets may be assigned (e.g., PROTEIN FAMILY) |
| 4 | Multiple homologous protein targets may be assigned (e.g., PROTEIN FAMILY) |
| 3 | Target assigned is molecular non-protein target |
| 1 | Target assigned is non-molecular |
| 0 | Default value - Target assignment has yet to be curated |

### Recommended Usage

- **High confidence (8-9)**: Best for focused target analysis and SAR studies
- **Medium confidence (6-7)**: Good balance of data quantity and quality
- **Lower confidence (0-5)**: Useful for large-scale analyses, but may include less specific target assignments

**Note on ADMET data**: Confidence scores reflect target assignment certainty, not measurement reliability. ADMET assays (permeability, clearance, solubility, etc.) typically have low confidence scores because they measure whole-cell or physicochemical properties rather than specific protein targets. A low confidence score for an ADMET assay does not indicate unreliable data - use `--confidence-scores 0,1,2,3,4,5,6,7,8,9` when retrieving ADMET data.

## Standard Relations and Censored Data

ChEMBL bioactivity measurements include a `standard_relation` field that indicates the relationship between the measured value and the reported concentration.

### Relation Types

- **`=`**: The measured value equals the reported concentration
- **`<`**: The compound is active at concentrations _below_ the reported value (_active_ at the reported concentration)
- **`<<`**: Stronger indication that the compound is active at concentrations _well below_ the reported value (_active_ at the reported concentration)
- **`>`**: The compound is active at concentrations _above_ the reported value (_inactive_ at the reported concentration)
- **`>>`**: Stronger indication that the compound is active at concentrations _well above_ the reported value (_inactive_ at the reported concentration)
- **`~`**: Approximate measurement (CAPRICHO handles these as ±0.5 log units)

### Working with Censored Data

**Important**: ChEMBL only pre-calculates pchembl_value for exact measurements (`standard_relation='='`). To include censored data (`<`, `>`), you *must* use the `--calculate-pchembl` flag:

```bash
# Default: Only includes exact measurements (=)
capricho get --target-ids CHEMBL203 --output-path egfr_exact.csv

# Include censored measurements: MUST use --calculate-pchembl
capricho get --target-ids CHEMBL203 \
  --standard-relation "=,<,>" \
  --calculate-pchembl \
  --output-path egfr_all.csv
```

Without `--calculate-pchembl`, you'll get an error if you request censored data, but the data will still be fetched:
```
ERROR: pchembl_values are only calculated for standard_relation='='.
If you want to use censored data, please set calculate_pchembl to True.
```

### Aggregation with Censored Data

When aggregating data with censored measurements, CAPRICHO only combines measurements that have:
1. Identical `standard_relation` values
2. In case of identical (`=`) standard_relation, statistics will be calculated for compound-target pairs with multiple exact measurements.
3. Censored measurements (`<`, `>`, `<<`, `>>`) are only combined with exact measurement matches (e.g.: < 6.0 will not be combined with < 5.0).

This conservative approach prevents mixing incompatible measurement types (e.g., averaging an exact value with a lower bound).

### pchembl_relation: Inverted Relations for -log Scale

When working with pChEMBL values ($-log_{10}(Molar)$), the direction of comparison operators is inverted compared to the original concentration values. CAPRICHO automatically creates a `pchembl_relation` column during binarization to make this relationship explicit:

**Relation Inversion Logic:**
- `standard_relation` `<` (low concentration, active) → `pchembl_relation` `>` (high pChEMBL, active)
- `standard_relation` `>` (high concentration, inactive) → `pchembl_relation` `<` (low pChEMBL, inactive)
- `standard_relation` `=` → `pchembl_relation` `=` (unchanged)
- `standard_relation` `~` → `pchembl_relation` `~` (unchanged)

**Example Interpretation:**
For a measurement with `IC50` = 1 µM (pChEMBL = 6.0) and `standard_relation` = `<`:
- Original: IC50 < 1 µM (active at concentrations below 1 µM)
- pChEMBL: pchembl_value > 6.0 (higher pChEMBL = more active)
- With threshold = 6.0: classified as **active (1)**

This inverted relation column is automatically added when you run the `binarize` command and helps interpret how measurements relate to activity thresholds on the -log scale.

### Activity Data Analysis

**For binary classification** (active/inactive), use the `binarize` command which properly handles censored measurements. Following the example above, we have:

```bash
capricho binarize -i egfr_all.csv -o egfr_binary.csv -t 6.0
```
See the CLI reference for detailed binarization options.

## Binarization Conflicts

When a compound-target pair has measurements from both exact (`=`) and censored (`<`, `>`) assays, the resulting binary labels may disagree. For example, an exact IC50 of 100 nM (pChEMBL 7.0) says "active", while a censored measurement `> 10 µM` (pChEMBL < 5.0) says "inactive".

### Why Conflicts Happen

The most common pattern (~97% of EGFR conflicts) is **exact vs censored**: an exact measurement classifies a compound differently from a censored bound. This often occurs when:

- A high-throughput screen reports a censored inactive result (`> 10 µM`)
- A focused follow-up study measures an exact IC50 well below the threshold
- Different assay formats have different dynamic ranges

### Conflict Resolution Strategies

By default, CAPRICHO flags conflicts but keeps all rows. Use `--conflict-resolution` to resolve them automatically:

**`relation`** — Trust exact measurements over censored bounds. This is the most scientifically grounded strategy: an exact IC50 measurement is inherently more informative than a "> 10 µM" bound.

**`majority`** — Measurement-level majority vote. Each row's pipe-separated raw values (e.g., `pchembl_value = "6.0|6.5|7.0"`) are split and each individual measurement is classified against the threshold. Every measurement gets one vote, so a row with 3 values contributes 3 votes. This is more granular than row-level counting: if a row's mean is active but one individual value is below the threshold, that value votes inactive. Falls back to dropping all rows on a tie.

**`confidence`** — Keep the row with the highest ChEMBL confidence score. Useful when assays vary in target assignment certainty.

**`drop`** — Remove all conflicting rows entirely. The most conservative option; guarantees no ambiguous labels remain.

### Conflict Severity

The conflict report classifies each conflict by severity based on measurement spread:

- **Low** (spread < 1.0 log units): Measurements are close to each other and the threshold
- **Medium** (spread 1.0–2.0 log units): Moderate disagreement
- **High** (spread > 2.0 log units): Large disagreement, likely different assay conditions or systematic error

### Choosing a Strategy

For most use cases, `relation` is a safe default — it keeps the most informative measurements. Use `majority` when you have many aggregated measurements and want each individual value to vote on the label. Use `drop` when label accuracy is critical and you'd rather lose data than risk mislabeling.

Generate a conflict report (`-rp conflicts.json`) to inspect the conflicts before deciding:

```bash
# Inspect conflicts first (no resolution)
capricho binarize -i data.csv -o binary.csv -t 7.0 -rp conflicts.json

# Then resolve with your chosen strategy
capricho binarize -i data.csv -o binary.csv -t 7.0 -cr relation -rp conflicts.json
```

### Post-Resolution Deduplication

When any conflict resolution strategy is active, CAPRICHO deduplicates the output to **one row per compound-target pair**. This produces ML-ready data where each compound-target combination has a single binary label.

During deduplication:

1. **Individual measurement filtering** — Within each row, the pipe-separated raw values (e.g., `pchembl_value = "5.5|6.5|7.0"`) are split and each measurement is classified against the threshold. Measurements that disagree with the row's resolved binary label are removed. All other pipe-separated columns (`assay_chembl_id`, `molecule_chembl_id`, etc.) are filtered at the same positions to stay aligned.

2. **Row merging** — Multiple rows for the same compound-target pair (e.g., one with `standard_relation = "="` and another with `standard_relation = "<"`) are merged into a single row. The `standard_relation` column becomes pipe-separated to reflect per-measurement relations (e.g., `"=|=|<"`).

3. **Stats recalculation** — `pchembl_value_mean`, `_std`, `_median`, and `_counts` are recalculated from the kept measurements only.

The resulting `pchembl_value` and `standard_relation` columns serve as a register tracing which source values compose the binarized label.

## Quality Control Filters

CAPRICHO provides multiple layers of quality control:

### Bioactivity Type Filtering
```bash
--bioactivity-type IC50,Ki,EC50
```
Focus on specific measurement types relevant to your analysis.

### Relation Filtering
```bash
# Only exact measurements (default behavior, pchembl pre-calculated by ChEMBL)
--standard-relation "="

# Include censored data (requires --calculate-pchembl)
--standard-relation "=,<,>" --calculate-pchembl
```
Choose measurement precision level based on your analysis needs.

### Date Requirements
```bash
--require-doc-date
```
Ensure all data has associated publication dates.

### Assay Size Constraints
```bash
--min-assay-size 10 --max-assay-size 1000
```
Filter assays by number of tested compounds.

## Reproducibility

CAPRICHO ensures full reproducibility through several mechanisms:

### Recipe Files

Every run generates a JSON recipe file containing the full command and all parameters used:

```json
{
  "command": "capricho get --target-ids CHEMBL203 --output-path egfr_data.csv",
  "capricho version": "0.1.0",
  "molecule_ids": [],
  "target_ids": ["CHEMBL203"],
  "assay_ids": [],
  "document_ids": [],
  "calculate_pchembl": false,
  "output_path": "egfr_data.csv",
  "confidence_scores": [7, 8, 9],
  "bioactivity_type": ["Potency", "Kd", "Ki", "IC50", "AC50", "EC50"],
  "standard_relation": ["="],
  "assay_types": ["B", "F"],
  "chembl_version": "36",
  "compound_equality": "connectivity",
  "value_column": "pchembl_value"
}
```

This allows exact reproduction of your data curation workflow.

### Version Control

Specify exact ChEMBL versions:
```bash
capricho get --target-ids CHEMBL203 --chembl-version 33
```

### Transparent Processing

All filtering steps are logged and flagged data is preserved for inspection.

## Output Structure

Understanding CAPRICHO's output helps you make the most of your curated data. Each `capricho get` run produces multiple files:

### Main Data File (`*_data.csv`)
- Aggregated bioactivity measurements
- Standardized column names
- Quality and processing flags in dedicated columns

### Recipe File (`*_recipe.json`)
- Complete record of all parameters used
- Enables exact reproduction of the workflow

### Pre-aggregation Data (`*_not_aggregated.csv`)
- Individual measurements before aggregation
- Useful for understanding how data was combined
- Can be skipped with `--skip-not-aggregated`

### Removed Subset (`*_removed_subset.csv`)
- Entries that were filtered out during curation
- Allows inspection of what was excluded and why

This multi-file approach ensures transparency while providing clean, analysis-ready data.