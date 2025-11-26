# Well Log Preprocessing Pipeline

This project builds a master, cleaned dataset from raw LAS well logs (GR, RT, DEN, NPHI, PEF, SONIC, etc.) for downstream ML tasks.

## Setup

From the `code` directory (with your `.venv` already created):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run preprocessing

From the `code` directory:

```bash
python -m src.preprocess_well_logs \
  --raw-folder "../dataset/raw" \
  --output "output/master_welllogs.csv"
```

Defaults (if you omit the arguments):
- `raw-folder` = `../dataset/raw`
- `output` = `output/master_welllogs.csv`

## What it does

1. Reads all `.las` files from the raw folder using `lasio`.
2. Extracts curves into a single pandas DataFrame and tags rows with `API_NUMBER` inferred from the filename.
3. Replaces LAS null values (`-999.25`) with `NaN` and linearly interpolates numeric columns.
4. Drops depth samples with more than 30% missing among the curves.
5. Min–max normalizes available curves among `[GR, RT, DEN, NPHI, PEF, SONIC]`.
6. Writes the cleaned, flat table to CSV for ML modeling.
