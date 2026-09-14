# CLI Reference

CAPRICHO provides five main commands: `download`, `explore`, `get`, `prepare`, and `binarize`. This section provides comprehensive documentation for all command-line options.

## capricho download

Downloads the ChEMBL SQL database using `chembl_downloader`.

```bash
capricho download [OPTIONS]
```

### Options

| Option | Description | Default |
|---|---|---|
| `--version`, `-v` | ChEMBL version to download | latest |
| `--prefix`, `-p` | Custom pystow storage path | `~/.data/chembl/` |
| `--set-from-path` | Use a ChEMBL SQLite database already on the system instead of downloading | `None` |
| `--unset-path` | Forget the database registered with `--set-from-path` | `False` |

### Examples

```bash
# Download latest ChEMBL version
capricho download

# Download specific version
capricho download --version 33

# Use custom storage location (this will install version 25 it on ~/.data/old-chembl)
capricho download --version 25 --prefix old-chembl/
```

### Using a database you already have

If you keep your own ChEMBL SQLite dumps, point CAPRICHO at them instead of downloading a
second copy. The database is read where it lies; nothing is moved or copied.

```bash
# Register a single release
capricho download --set-from-path /data/chembl/chembl_35.db

# Register every chembl_<version>.db found under a directory
capricho download --set-from-path /data/chembl/

# Register only one release out of a directory holding several
capricho download --set-from-path /data/chembl/ --version 35

# Go back to downloading that release
capricho download --unset-path --version 35
```

The release is read from the `chembl_<version>.db` file name; pass `--version` explicitly when
the file is named otherwise. Registration writes
`~/.data/chembl/chembl_downloader_config_<version>.json`, which every later command honours, so
`capricho get --chembl-version 35` and `capricho explore --version 35` read the registered file.

Before registering, CAPRICHO checks that the file is a ChEMBL database and that the release it
states for itself — every dump records this in its `version` table — is the release it is being
registered under. Registering a ChEMBL 36 dump as ChEMBL 37, whether through a mistyped
`--version` or a renamed file, is therefore refused rather than silently misreporting the
provenance of every dataset drawn from it. A database that states no release is registered on
the strength of its file name, with a warning. Each run reports the database it reads, and a
registered database that is later moved or deleted fails with an actionable message.

## capricho explore

Explore the downloaded ChEMBL SQL database.

```bash
capricho explore [OPTIONS]
```

### Options

| Option | Description |
|---|---|
| `--version`, `-v` | ChEMBL version to use (defaults to latest) |
| `--list-tables`, `-list` | List all tables within the SQL database and exit |
| `--table`, `-t` | Explore a specific table |
| `--search-column`, `-search` | Search for tables containing a column name pattern |
| `--query`, `-q` | Run a custom SQL query |
| `--format`, `-f` | Console output format for tables (`markdown` or `csv`) |
| `--output`, `-o` | Save primary result DataFrame to file (format inferred from extension) |
| `--colorize` / `--no-colorize` | ANSI color cycling on console table rows |

### Examples

```bash
# List all available tables on the relational database
capricho explore --list-tables

# Examine the activities table
capricho explore --table activities

# Find tables with 'pchembl' columns
capricho explore --search-column pchembl

# Check available standard_relation types in ChEMBL
capricho explore --query "SELECT standard_relation, COUNT(*) as count FROM activities WHERE standard_relation IS NOT NULL GROUP BY standard_relation ORDER BY count DESC"
```

### Useful Queries for Data Exploration

These queries can help you understand the data before running `capricho get`:

**Count activities by bioactivity type:**
```bash
capricho explore --query "SELECT standard_type, COUNT(*) as count FROM activities WHERE standard_type IS NOT NULL GROUP BY standard_type ORDER BY count DESC LIMIT 20"
```

**Count activities by assay type:**
```bash
capricho explore --query "SELECT assay_type, COUNT(*) as count FROM assays GROUP BY assay_type ORDER BY count DESC"
```

**Check confidence score distribution:**
```bash
capricho explore --query "SELECT confidence_score, COUNT(*) as count FROM assays GROUP BY confidence_score ORDER BY confidence_score DESC"
```

**Check all standard units:**
```bash
capricho explore --query "SELECT standard_units, COUNT(*) as count FROM activities WHERE standard_units IS NOT NULL GROUP BY standard_units ORDER BY count DESC LIMIT 30"
```

## capricho get

Filter, download, and process bioactivity data from ChEMBL. This is the main command of CAPRICHO.

```bash
capricho get [OPTIONS]
```

### Input ID Options

Specify which ChEMBL entities to retrieve data for:

| Option | Description | Default |
|---|---|---|
| `-mids`, `--molecule-ids` | ChEMBL molecule IDs, comma-separated | `[]` |
| `-tids`, `--target-ids` | ChEMBL target IDs, comma-separated | `[]` |
| `-asids`, `--assay-ids` | ChEMBL assay IDs, comma-separated | `[]` |
| `-dids`, `--document-ids` | ChEMBL document IDs, comma-separated | `[]` |

### Filtering Options

Control which bioactivity data to include:

| Option | Description | Default |
|---|---|---|
| `-c`, `--confidence-scores` | Confidence scores to filter, comma-separated | `[7, 8, 9]` |
| `-biotype`, `--bioactivity-type` | Bioactivity types to filter, comma-separated | `['Potency', 'Kd', 'Ki', 'IC50', 'AC50', 'EC50']` |
| `-rel`, `--standard-relation` | Filter by standard relation (`=`, `<`, `>`, `~`), comma-separated. **Note:** Including `<` or `>` requires `--calculate-pchembl`. See [Standard Relations](concepts.md). | `['=']` |
| `-units`, `--standard-units` | Filter by standard units, comma-separated. Useful for ADMET data with specific units like `%` (percent inhibition). | `None` |
| `-at`, `--assay-types` | Assay types (B, F, A, T, P), comma-separated | `['B', 'F']` |
| `-cr`, `--chembl-release` | Only fetch data reported **up to** a certain ChEMBL release | `None` |
| `-reqdoc`, `--require-doc-date` | Filter out bioactivities without a document date | `False` |
| `-maxas`, `--max-assay-size` | Maximum number of compounds in an assay | `None` |
| `-minas`, `--min-assay-size` | Minimum number of compounds in an assay | `None` |
| `-maso`, `--min-assay-overlap` | Minimum overlapping compounds between assays | `0` |

#### Confidence Scores
ChEMBL assigns confidence scores from 0-9 based on target assignment certainty:
- **9**: Direct single protein target assigned
- **8**: Homologous single protein target assigned
- **7**: Direct protein complex subunits assigned
- **6**: Homologous protein complex subunits assigned
- **5**: Multiple direct protein targets may be assigned (e.g., PROTEIN FAMILY)
- **4**: Multiple homologous protein targets may be assigned (e.g., PROTEIN FAMILY)
- **3**: Target assigned is molecular non-protein target
- **1**: Target assigned is non-molecular
- **0**: Default value - Target assignment has yet to be curated

#### Bioactivity Types
Common bioactivity measurements:
- **IC50**: Half maximal inhibitory concentration
- **EC50**: Half maximal effective concentration
- **Ki**: Inhibition constant
- **Kd**: Dissociation constant
- **AC50**: Half maximal activity concentration
- **Potency**: General potency measurement (see `assay_description` for more detail)

#### Assay Types
- **B**: Binding assay
- **F**: Functional assay
- **A**: ADMET assay
- **T**: Toxicity assay
- **P**: Physicochemical assay
- **U**: Unclassified

Further information on assay types and confidence scores can be found in the [ChEMBL documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions).

### Processing & Aggregation Options

Control how data is processed and aggregated:

| Option | Description | Default |
|---|---|---|
| `-calc`, `--calculate-pchembl` | Calculate pChEMBL values if not reported. **Required when using censored data** (`--standard-relation` includes `<` or `>`). See [Standard Relations](concepts.md). | `False` |
| `-vcol`, `--value-column` | Column holding the experimental measurement to summarize (mean/median/std). Use `standard_value` for non-pChEMBL data (e.g., ADMET assays with % inhibition). See [Non-pChEMBL Aggregation](non-pchembl-aggregation). | `pchembl_value` |
| `-conu`, `--convert-units` | Convert units to standard formats before aggregation. See [Unit Conversion](unit-conversion). | `False` |
| `-chiral`, `--chirality` | Consider chirality during fingerprint calculation | `False` |
| `-duchi`, `--drop-unassigned-chiral` | Drop entries with unassigned chiral centers | `False` |
| `-cure`, `--curate-annotation-errors` | Apply curation for pChEMBL annotation errors | `False` |
| `-mutagg`, `--aggregate-mutants` | Aggregate data on targets regardless of mutation | `False` |
| `-smr`, `--strict-mutant-removal` | Flag assays with mutant-related keywords for removal | `False` |
| `-cpd-eq`, `--compound-equality` | Method for compound equality determination | `connectivity` |
| `-mcols`, `--metadata-columns` | Extra metadata columns to keep, comma-separated | `[]` |
| `-idcols`, `--id-columns` | Additional columns to append to the aggregation key (compound + task), comma-separated. E.g. `assay_chembl_id` keeps measurements from different assays separate. | `[]` |

#### Aggregated output diagnostics

The aggregated output includes `shared_identifier_group`. Rows sharing the selected
compound identifier and `target_chembl_id` receive the same label when they were kept
separate by mutation, `--id-columns`, or another preserved readout field; all other rows
contain `NaN`. The column is diagnostic metadata, not a source-data quality flag.
`data.shared_identifier_group.value_counts()` reports each group's size. With the default
compound-equality method the identifier is `connectivity`; the message and grouping use `inchi`,
`inchikey`, or `smiles` when one of those methods is selected.

#### Aggregation Column Options
- **pchembl_value**: (Default) Aggregate on pChEMBL values (-log10 molar potency). Uses geometric mean.
- **standard_value**: Aggregate on raw standard_value column. Uses arithmetic mean. Useful for ADMET data with non-molar units (%, permeability, etc.).

#### Compound Equality Methods
- **connectivity**: (Default) Uses the first InChIKey block, ignoring stereochemistry
- **inchi**: Uses the complete standard InChI, including specified stereochemistry
- **inchikey**: Uses the complete 27-character InChIKey
- **mixed_fp**: Uses ECFP4 and RDKit fingerprints (each with 2048 bits) for identity determination
- **smiles**: Uses standardized SMILES strings directly for exact string matching

#### Useful Metadata Columns
- `organism`: Source organism
- `tissue`: Tissue type
- `cell_type`: Cell line information
- `assay_description`: Detailed assay description
- `target_type`: Type of target (e.g., SINGLE PROTEIN, PROTEIN FAMILY)

### Output & Backend Options

Control output format and data source:

| Option | Description | Default |
|---|---|---|
| `-o`, `--output-path` | Path to save the output files | `chembl_data.csv` |
| `-skip-agg`, `--skip-not-aggregated` | Skip saving pre-aggregation data | `False` |
| `-rec`, `--skip-recipe` | Skip saving the JSON recipe file | `False` |
| `-back`, `--chembl-backend` | Backend to use for ChEMBL interaction | `downloader` |
| `-v`, `--chembl-version` | ChEMBL version used by `chembl_downloader` | `None` |

#### Backend Options
- **downloader**: (Default) Uses local SQL database, faster for large queries
- **webresource**: Uses ChEMBL web API, no local download required

## Complete Examples

### Basic Target Analysis
```bash
capricho get --target-ids CHEMBL203 --output-path egfr_analysis.csv
```

### High-Quality Multi-Target Dataset
```bash
capricho get \
  --target-ids CHEMBL203,CHEMBL204,CHEMBL279 \
  --confidence-scores 8,9 \
  --bioactivity-type IC50,Ki \
  --standard-relation "=" \
  --aggregate-mutants \
  --metadata-columns organism,tissue \
  --output-path high_quality_dataset.csv
```

### ADMET Data with Unit Conversion

Retrieve Caco-2 permeability data with unit conversion and aggregation on `standard_value`:

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
- Fetches data from specific Caco-2 permeability assays
- Uses ADMET assay type (`-at A`)
- Aggregates on `standard_value` instead of pChEMBL (permeability isn't a potency measurement)
- Converts permeability units to a common format (`10^-6 cm/s`)
- Groups by `standard_units` and `assay_cell_type` during aggregation

## capricho prepare

Clean aggregated bioactivity data by filtering entries based on quality and processing flags introduced during `capricho get`. Optionally, transform the cleaned data into a multitask activity matrix where rows are compounds and columns are tasks (e.g., targets).

```bash
capricho prepare [OPTIONS]
```

### Required Options

| Option | Description |
|---|---|
| `-i`, `--input-path` | Path to aggregated data file (CSV, TSV, or Parquet) |
| `-o`, `--output-path` | Path to save the output file |

### Quality Flag Filtering Options

These flags remove entries with specific quality concerns. Each flag corresponds to a comment added during `capricho get`:

| Option | Description | Default |
|---|---|---|
| `--drop-undefined-stereo` | Drop entries with undefined stereochemistry | `False` |
| `--drop-potential-duplicate` | Drop entries flagged as potential duplicates across documents | `False` |
| `--drop-data-validity` | Drop entries with data validity comments from ChEMBL | `False` |
| `--drop-unit-error` | Drop entries with unit annotation errors (3.0 or 6.0 log unit differences) | `False` |
| `--drop-mixture` | Drop entries containing mixtures in SMILES | `False` |
| `--drop-activity-comment` | Drop entries whose `activity_comment` reports inactivity while the source `standard_relation` is `=` | `False` |
| `--drop-assay-size` | Drop entries outside assay size bounds (both too small and too large) | `False` |
| `--drop-insufficient-overlap` | Drop entries from assays with insufficient compound overlap | `False` |
| `--remove-flags` | Custom quality flags to remove, comma-separated. Rows with these flags in `data_dropping_comment` will be filtered out. | `None` |

### Data Cleaning Options

| Option | Description | Default |
|---|---|---|
| `--deduplicate` | Remove duplicate pChEMBL values within aggregated rows and recalculate statistics | `False` |
| `--resolve-annotation-error` | Resolve unit annotation errors by keeping measurement from earliest document. Use `first` to enable. | `None` |

### Activity Matrix Options

These options control the optional multitask activity matrix output:

| Option | Description | Default |
|---|---|---|
| `--task-col` | Column to use as task identifier | `target_chembl_id` |
| `--compound-col` | Compound identifier column (`connectivity`, `inchi`, `inchikey`, or `smiles`) | `connectivity` |
| `--smiles-col` | Column containing SMILES strings | `smiles` |
| `-vcol`, `--value-column` | Column holding the experimental measurement, as passed to `capricho get --value-column`. Statistics are read from `{value_column}_mean`. | `pchembl_value` |
| `--id-columns` | Additional columns to combine with `task_col` for composite task identifiers. Use the same columns passed to `capricho get --id-columns` during aggregation. | `None` |

Use the compound identifier selected during aggregation when creating the matrix. For example,
a dataset retrieved with `--compound-equality inchikey` should normally be prepared with
`--compound-col inchikey`.

### Output Options

| Option | Description | Default |
|---|---|---|
| `--plot` | Path to save comparability plots (e.g., `comparability.png`). If not provided, no plot is generated. | `None` |

### Understanding Quality Flags

Quality flags are added to the `data_dropping_comment` column during `capricho get`. Common flags include:

- **Undefined stereochemistry**: Compounds with unassigned chiral centers
- **Potential duplicate**: Quality flag introduced by the ChEMBL team, indicting that compound-target pair reported in multiple documents with identical values
- **Data validity comment**: ChEMBL's own data quality annotations
- **Unit annotation error**: Measurements differing by exactly 3.0 or 6.0 log units (suggesting unit conversion errors)
- **Mixture in SMILES**: SMILES containing multiple components (`.` separator)
- **Activity with exact standard relation and inactivity-like comment**: An inactivity-like `activity_comment` occurs with `standard_relation = '='`. CAPRICHO flags the combination for review while preserving the source comment, relation, and value.
- **Assay size too small/large**: Assays outside the specified size bounds
- **Insufficient assay overlap**: Assays without enough shared compounds for reliable comparison

### Output Files

The prepare command generates two files:

- **Prepared data** (`*_prepared.csv`): The cleaned data after filtering quality flags
- **Activity matrix** (specified by `-o`): Rows are compounds (indexed by `compound_col`), columns are tasks, plus a `smiles` column. Suitable for multitask ML models.

### Examples

#### Basic Preparation
```bash
# Clean data and output activity matrix
capricho prepare -i egfr_data.csv -o egfr_matrix.csv
```

#### Filtering Quality Flags
```bash
# Remove potential duplicates and data validity issues
capricho prepare -i egfr_data.csv -o egfr_clean.csv \
    --drop-potential-duplicate \
    --drop-data-validity
```

#### Strict Quality Filtering
```bash
# Apply multiple quality filters for high-confidence data
capricho prepare -i kinase_data.csv -o kinase_clean.csv \
    --drop-potential-duplicate \
    --drop-data-validity \
    --drop-unit-error \
    --drop-undefined-stereo
```

#### With Deduplication
```bash
# Remove duplicate values and recalculate statistics
capricho prepare -i data.csv -o clean_data.csv \
    --deduplicate \
    --drop-potential-duplicate
```

#### Resolve Annotation Errors
```bash
# Keep earliest measurement when annotation errors are detected
capricho prepare -i data.csv -o clean_data.csv \
    --resolve-annotation-error first \
    --drop-unit-error
```

#### Generate Comparability Plots
```bash
# Output plots showing data comparability across assays
capricho prepare -i data.csv -o clean_data.csv \
    --drop-potential-duplicate \
    --plot comparability.png
```

This generates two plots:
- `comparability_cleaned.png`: Comparability of data after filtering
- `comparability_flags.png`: Multi-panel view showing remaining flags

#### Composite Task Identifiers
```bash
# Use id-columns if data was aggregated with --id-columns
capricho prepare -i data.csv -o matrix.csv \
    --id-columns assay_cell_type,standard_units
```

#### Custom Flag Removal
```bash
# Remove entries with custom flags
capricho prepare -i data.csv -o clean_data.csv \
    --remove-flags "Censored activity comment,Mutant assay"
```

## capricho binarize

Convert aggregated bioactivity data to binary labels (active/inactive) based on a pChEMBL threshold. This command handles censored measurements (< and >) and validates agreement between different measurement types.

```bash
capricho binarize [OPTIONS]
```

### Required Options

| Option | Description |
|---|---|
| `-i`, `--input-path` | Path to aggregated data file (CSV, TSV, or Parquet) |
| `-o`, `--output-path` | Path to save the binarized output file |

### Binarization Options

| Option | Description | Default |
|---|---|---|
| `-t`, `--threshold` | Activity threshold for binarization (pChEMBL scale) | `6.0` (1 µM) |
| `-vcol`, `--value-column` | Column to use for binarization | `pchembl_value_mean` |
| `-cid`, `--compound-id-col` | Column name for compound identifiers | `connectivity` |
| `-tid`, `--target-id-col` | Column name for target identifiers | `target_chembl_id` |
| `-rel`, `--relation-col` | Column name for standard_relation values | `standard_relation` |
| `-bcol`, `--binary-col` | Name for the output binary column | `activity_binary` |
| `-cmp-mut`, `--compare-across-mutants` | Compare measurements across mutants for conflicts | `False` |
| `-rp`, `--conflict-report-path` | Path to save detailed conflict report as JSON | `None` |
| `-cr`, `--conflict-resolution` | Strategy for resolving conflicts: `drop`, `relation`, `confidence`, `majority` | `None` |

### Understanding pChEMBL Thresholds

The pChEMBL scale is -log10(Molar), where higher values indicate higher activity:

- **pChEMBL 6.0** = 1 µM (common threshold)
- **pChEMBL 6.5** = 316 nM
- **pChEMBL 7.0** = 100 nM (stringent threshold)
- **pChEMBL 5.0** = 10 µM (permissive threshold)

### Handling Standard Relations

The binarization process handles different measurement types:

- **`=`** (discrete): Direct comparison to threshold
- **`~`** (approximate): Uses lower bound for conservative classification (±0.5 log units)
- **`<`, `<<`** (censored active): Compound is _more_ active than reported value
- **`>`, `>>`** (censored inactive): Compound is _less_ active than reported value

### Conflict Detection and Resolution

The command flags measurements that disagree for the same compound-target pair:

- **Mixed discrete/censored conflicts**: When discrete measurements (`=`, `~`) disagree with censored measurements (`<`, `>`)
- **Binary label conflicts**: When measurements result in different activity classifications (active vs inactive)
- **Mutation handling**: Use `--compare-across-mutants` to control whether different mutants are compared

#### Conflict Resolution Strategies

By default, conflicts are only flagged in `data_dropping_comment`. Use `--conflict-resolution` to automatically resolve them:

| Strategy | Behavior | Fallback |
|----------|----------|----------|
| `drop` | Remove all rows for conflicting pairs | -- |
| `relation` | Keep exact (`=`) rows, drop censored | Drop all if no `=` exists |
| `confidence` | Keep row with highest `confidence_score` | Drop all on tie |
| `majority` | Classify each individual measurement against the threshold; majority label wins | Drop all on tie |

The `majority` strategy splits the pipe-separated raw values (e.g., `pchembl_value = "6.0|6.5|7.0"`) and classifies each individual measurement against the threshold. Each measurement gets one vote. This means a row whose mean is above the threshold but contains individual values below it will contribute some inactive votes. When the raw value column is not present, falls back to count-weighted or row-based voting.

```bash
# Resolve conflicts by keeping exact measurements
capricho binarize -i data.csv -o binary.csv -t 7.0 -cr relation

# Resolve by measurement-weighted majority vote, save report
capricho binarize -i data.csv -o binary.csv -t 7.0 -cr majority -rp conflicts.json
```

#### Conflict Report

Use `-rp` / `--conflict-report-path` to save a JSON report with:

- **Summary**: Total conflicts, conflict patterns (exact vs censored), active/inactive counts, MCC, resolution summary
- **Per-conflict details**: Measurements, vote summary, severity (low/medium/high based on measurement spread), recommendation, resolution outcome

#### Compound Identifiers for Conflict Detection

By default, conflicts are detected using the `connectivity` column, which groups compounds by
their molecular graph while ignoring stereochemistry. You can instead select any compound
identifier present in the aggregated output:

- **`connectivity`** (default): First InChIKey block; ignores stereochemistry
- **`inchi`**: Complete standard InChI
- **`inchikey`**: Complete 27-character InChIKey
- **`smiles`**: Standardized SMILES

For example, use full InChIKey identity with:

```bash
capricho binarize -i data.csv -o output.csv -cid inchikey
```

Use the same identity level chosen for aggregation unless you deliberately want to inspect
conflicts at a broader or narrower level.

### Examples

#### Basic Binarization
```bash
# Default threshold of 6.0 (1 µM)
capricho binarize -i aggregated_data.csv -o binarized_data.csv
```

#### Custom Threshold
```bash
# Stringent threshold of 7.0 (100 nM)
capricho binarize -i egfr_data.csv -o egfr_binary.csv -t 7.0
```

#### Using Median Values
```bash
# Use median instead of mean for binarization
capricho binarize \
  -i aggregated_data.csv \
  -o binary_median.csv \
  -vcol pchembl_value_median \
  -t 6.5
```

#### Compare Across Mutants
```bash
# Flag conflicts even when measurements are on different mutants
capricho binarize \
  -i kinase_data.csv \
  -o kinase_binary.csv \
  -t 6.5 \
  --compare-across-mutants
```

#### Resolve Conflicts and Generate Report
```bash
# Keep exact measurements, drop censored when they conflict
capricho binarize \
  -i egfr_data.csv \
  -o egfr_binary.csv \
  -t 7.0 \
  -cr relation \
  -rp conflict_report.json

# Measurement-weighted majority vote
capricho binarize \
  -i egfr_data.csv \
  -o egfr_binary.csv \
  -t 7.0 \
  -cr majority \
  -rp conflict_report.json
```

### Understanding pchembl_relation

The output file includes a `pchembl_relation` column that adjusts the standard_relation signs for the -log scale used in pChEMBL values. This makes it easier to interpret activity thresholds:

**Example:** For threshold = 6.0 (1 µM)
- `IC50` with `pchembl_value` = 6.0 and `pchembl_relation` = `>`
  - -log10[IC50 concentration] > 6.0
  - IC50 concentration < 1 µM
  - **active (1)**

The relation inversion happens because pChEMBL values are negative logarithms:
- `standard_relation` `<` (low concentration) → `pchembl_relation` `>` (high pChEMBL, active)
- `standard_relation` `>` (high concentration) → `pchembl_relation` `<` (low pChEMBL, inactive)
- `standard_relation` `=` → `pchembl_relation` `=` (unchanged)

This column is automatically generated during binarization and helps interpret the relationship between measurements and the activity threshold.

### Output Format

The output file contains all original columns plus:

- **Binary activity column** (default: `activity_binary`): 0 (inactive), 1 (active), or null (missing)
- **pchembl_relation column**: Standard relation adjusted for -log scale (see above)
- **Conflict flags**: Rows with disagreeing measurements are flagged in the `data_dropping_comment` column

Conflicting measurements are logged with detailed information about the disagreement.

#### Post-Resolution Deduplication

When a conflict resolution strategy is active (`-cr`), compound-target pairs are deduplicated to **one row per pair**. During deduplication:

- Individual measurements that disagree with the resolved binary label are filtered out from all pipe-separated columns
- Rows for the same compound-target pair are merged, concatenating their source values
- The `standard_relation` column becomes pipe-separated to match per-measurement relations (e.g., `"=|=|<"`)
- Statistics (`*_mean`, `*_std`, `*_median`, `*_counts`) are recalculated from the kept measurements only

The resulting `pchembl_value` and `standard_relation` columns serve as a register of source values that compose the binarized label.