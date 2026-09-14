# Working with ADMET Data

This tutorial covers retrieving and aggregating ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) data from ChEMBL using CAPRICHO.

## Why ADMET Data is Different

ADMET assays often report measurements in units that can't be converted to pChEMBL values:

- **Permeability**: cm/s, nm/s, 10^-6 cm/s
- **Clearance**: mL/min/kg
- **Half-life**: hours, minutes
- **Solubility**: µg/mL, mg/L
- **Percent inhibition**: %

These require different handling than traditional potency measurements (IC50, Ki, etc.).

## Step 1: Find Relevant Assays

First, explore ChEMBL to find ADMET assays of interest. For example, to find Caco-2 permeability assays:

```bash
capricho explore --query "
  SELECT assay_chembl_id, assay_description, COUNT(*) as n_activities
  FROM assays a
  JOIN activities act ON a.assay_id = act.assay_id
  WHERE a.assay_type = 'A'
    AND LOWER(assay_description) LIKE '%caco%'
    AND LOWER(assay_description) LIKE '%permeab%'
  GROUP BY assay_chembl_id
  ORDER BY n_activities DESC
  LIMIT 20
"
```

To see what units are used:

```bash
capricho explore --query "
  SELECT standard_units, COUNT(*) as count
  FROM activities act
  JOIN assays a ON act.assay_id = a.assay_id
  WHERE a.assay_type = 'A'
    AND LOWER(a.assay_description) LIKE '%caco%'
  GROUP BY standard_units
  ORDER BY count DESC
"
```

## Step 2: Basic ADMET Data Retrieval

Once you've identified assays, retrieve the data using `--value-column standard_value`:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
  --value-column standard_value \
  --output-path caco2_basic.csv
```

Key options:
- `--assay-types A`: Filter to ADMET assays
- `--confidence-scores 0,...,9`: ADMET assays not necessarily require a target to be assigned. Lower confidence scores can be acceptable.
- `--value-column standard_value`: Summarize raw measured values instead of pChEMBL

## Step 3: Handling Unit Heterogeneity

### Option A: Group by Units

To keep measurements with different units separate, use `--id-columns`:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --value-column standard_value \
  --id-columns standard_units \
  --output-path caco2_grouped_by_units.csv
```

This creates separate entries for the same compound if measured in different units.

### Option B: Convert Units

To convert all measurements to a common unit, use `--convert-units`:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --value-column standard_value \
  --convert-units \
  --output-path caco2_converted.csv
```

This converts permeability to `10^-6 cm/s`, enabling aggregation across assays.

### Option C: Filter by Units

To only include measurements with specific units, use `--standard-units`:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --standard-units "10^-6 cm/s,cm/s" \
  --value-column standard_value \
  --output-path caco2_filtered.csv
```

## Step 4: Maintaining Biological Context

ADMET measurements are highly dependent on experimental conditions. Use `--id-columns` to prevent aggregation across different conditions:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278 \
  --assay-types A \
  --value-column standard_value \
  --convert-units \
  --id-columns assay_cell_type,standard_units \
  --metadata-columns assay_description,assay_organism \
  --output-path caco2_with_context.csv
```

Common ID columns for ADMET data:
- `assay_cell_type`: Cell line used (e.g., Caco-2, MDCK)
- `standard_units`: Unit of measurement
- `assay_organism`: Species
- `assay_tissue`: Tissue source

## Example: Complete Caco-2 Dataset

Here's a comprehensive command for Caco-2 permeability data:

```bash
capricho get \
  --assay-ids CHEMBL1112933,CHEMBL3529279,CHEMBL3529278,CHEMBL3529277,CHEMBL3529276 \
  --assay-types A \
  --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
  --value-column standard_value \
  --convert-units \
  --id-columns standard_units,assay_cell_type \
  --metadata-columns assay_description,assay_organism \
  --drop-unassigned-chiral \
  --output-path caco2_permeability.csv
```

## Understanding the Output

When using `--value-column standard_value`, the output columns include:

| Column | Description |
|--------|-------------|
| `standard_value_mean` | Arithmetic mean of measurements |
| `standard_value_std` | Standard deviation |
| `standard_value_median` | Median value |
| `standard_value_n` | Number of measurements aggregated |
| `standard_units` | Unit of measurement (converted if `--convert-units` used) |

### Repeated activity identifiers in ADMET data

Different ADMET conditions can produce separate output rows with the same compound and task
identifiers. CAPRICHO keeps those rows separate when an ID column differs. With the current
default identity method, the concrete identifier columns are `connectivity` and
`target_chembl_id`. If `inchi`, `inchikey`, or `smiles` is selected through
`--compound-equality`, the diagnostic uses and names that compound identifier instead.

In the MDCK-MDR1 A→B permeability example, a current run reports:

```text
CAPRICHO found 10 separate activity rows with 5 repeated `connectivity` +
`target_chembl_id` combinations; varying fields: `standard_units`,
`assay_cell_type`.
```

One combination differs by reported unit (`10'-6/cm` versus `10^-6 cm/s`); the other four
differ by cell type (for example, `MDCK` versus `MDCK-MDR1`). The directional Caco-2 outputs
do not contain repeated combinations and therefore do not emit this message.

Inspect the affected rows and group sizes with:

```python
data[data["shared_identifier_group"].notna()]
data["shared_identifier_group"].value_counts()
```

If these conditions should remain separate downstream, use the fields named by the message—for
this example, `capricho prepare --id-columns standard_units,assay_cell_type`. Otherwise, resolve
the combinations deliberately before modeling.

## Working with Percent Inhibition Data

For percent inhibition assays (e.g., plasma protein binding):

```bash
capricho get \
    --assay-ids CHEMBL1924276,CHEMBL3388193,CHEMBL3388194,CHEMBL3388195,CHEMBL3388196,CHEMBL3091163,CHEMBL4325182,CHEMBL4420301 \
    --standard-units "%" \
    --assay-types A \
    --confidence-scores 0,1,2,3,4,5,6,7,8,9 \
    --value-column standard_value \
    --id-columns standard_units \
    -o percent_inhibition_data.csv
```

## Tips for ADMET Data

1. **Always check units first**: Use `capricho explore` to understand unit distribution before retrieval
2. **Use `--id-columns` liberally**: ADMET data is sensitive to experimental conditions
3. **Lower confidence scores are normal**: ADMET assays often have confidence scores < 7
4. **Verify conversions**: Check the `data_processing_comment` column to see which measurements were converted
5. **Consider biological relevance**: Not all measurements should be aggregated, even if units match

## Next Steps

- See [Concepts](../concepts.md) for details on unit conversion and aggregation
- See [CLI Reference](../cli-reference.md) for all available options
