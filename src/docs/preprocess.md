You’re right: “one operation fits all” is wrong for well logs. A good EDA must tell you *how this specific LAS collection behaves* so you can tune cleaning, scaling, feature selection, and even the ML task definition.

Below is a **robust, implementation-ready EDA workflow** for your LAS wells. If an agent follows this literally and inspects the outputs at each step, they will know how to configure your preprocessing pipeline for ML. The workflow assumes Python with `lasio`, `pandas`, `numpy`, `matplotlib`, `seaborn`.[1][2][3][4]

***

## 1. LAS ingestion and metadata inventory

Goal: understand **what you actually have** across all wells before touching ML.

### 1.1. Parse all LAS files and collect basic metadata

For each `.las` file:

- Extract:
  - Well-level info: API, well name, operator, county, field, UWI, etc.
  - Curve list: mnemonic, unit, description.
  - Depth range: min depth, max depth, step, direction (increasing/decreasing).
  - Null value flag (often `-999.25` or `-999.0`).
- Store in a **metadata table** with one row per well and one set of columns per property.[2][3]

Key questions this answers:

- How many wells total?
- What **curve types** and **units** are present and how consistently?
- How many wells have only GR vs GR+RT vs GR+DEN, etc.?

This informs:
- Which ML targets are even possible.
- Whether you build one model per curve family or a “common denominator” model.

***

## 2. Curve availability and consistency analysis

Goal: find **groups of wells with comparable curves** and **standardize mnemonics/units**.

### 2.1. Curve presence matrix

Build a matrix:

- Rows: wells.
- Columns: curve mnemonics (e.g. GR, RHOB, NPHI, RT, DT, PEF).
- Values: 1 if curve present in that well (non-null span > threshold), else 0.

From this you can:

- Identify the **most common curve combinations** (e.g. GR only, GR+RT, GR+DEN).
- Decide on:
  - **Core feature set**: curves that exist in enough wells to matter.
  - **Secondary sets**: curves for specialized sub-experiments.

If, for example, you find:

- GR present in 95% of wells.
- Density in 3%.
- Sonic in 1%.

Then:

- Main project: GR-based modeling (denoising/gap-filling/lithology proxies).
- Small case-study: GR→RHOB or GR→DT on the very few wells with those curves.[5][6][7]

### 2.2. Mnemonic harmonization

Logs for the same physical quantity may have different mnemonics (e.g. GR, GR_R25, SGR; RHOB, RHOZ; NPHI, NPHI_S, NPHI_C).

- Build a mapping: raw mnemonic → standardized name (e.g. any “G”*R* → `GR`).
- Do this by:
  - Inspecting curve descriptions and units in `.las.curves`.
  - Manually defining a few mapping rules.
- Apply mapping to create consistent columns.

This determines **whether you can pool wells for ML** or must treat some as separate types.

***

## 3. Depth grid and sampling quality

Goal: understand **depth sampling**, **gaps**, and **overlaps**, because ML models assume some regularity.

### 3.1. Depth direction and step

For each well:

- Check:
  - Is depth monotonically increasing or decreasing?
  - What is the **mode depth step** (median difference between consecutive depth samples)?
  - Are there irregular step jumps?

Produce:

- `DEPTH_MIN`, `DEPTH_MAX`, `DEPTH_STEP_MODE`, `DEPTH_DIRECTION` for each well.
- Histograms of depth ranges and step sizes across wells.

This informs:

- Whether you need to **resample each well to a common depth grid** (e.g. 0.5 ft or 0.1524 m) for models that expect equal-length sequences.[8][9]
- Whether some wells should be dropped due to extremely irregular sampling.

### 3.2. Valid logging interval per well

For each curve in each well:

- Identify:
  - Top and bottom of valid logging (where curve is non-null).
  - Where nulls or missing segments occur.

Summaries:

- Depth percentage of non-null for each curve per well.
- Wells with too short valid intervals (e.g. <500 ft) can be excluded or handled differently.

This helps define **masking strategy** for gap-filling and lets you pick clean wells for ML training.

***

## 4. Value distributions and physical sanity checks

Goal: check **ranges, units, outliers, and physical plausibility**.

For each standard curve family (e.g. GR, RT, RHOB, NPHI, DT):

### 4.1. Global distribution per curve type

Across all wells (after dropping nulls):

- Plot:
  - Histograms and KDEs for each log (GR, RT, etc.).
  - Boxplots per curve type, optionally grouped by well or by “well class” (e.g. oil/gas).[10][1]
- Compute:
  - Min, max, mean, median.
  - Standard deviation, skewness, kurtosis.
  - 1st, 25th, 75th, 99th percentiles.

Compare to **plausible ranges**:

- GR: ~0–250 API; typical reservoir: 20–150.
- Deep RT: 0.2–2000 ohm·m (log10 transform).
- RHOB: 1.9–2.9 g/cc.
- NPHI: 0–0.6 fraction.
- DT: 40–200 μs/ft.

If you see values far outside, mark them as **suspect** and define rules for:

- Hard clipping (e.g. GR < 0 or > 300 → null).
- Winsorization at, say, 1st and 99th percentile.
- Depth-specific weirdness (entire sections with crazy values → mark as invalid).[8][10]

### 4.2. Per-well distribution comparison

For each curve:

- Plot one boxplot per well (x-axis: well, y-axis: log values), or violin plots grouped by well.

This reveals:

- Wells with systematically shifted distributions (e.g. one GR curve scaled incorrectly, or in different units).
- Candidates for **well-level normalization** or **exclusion**.

Outcome: define **per-curve cleaning policy** (clip ranges, null codes) and decide on **global vs per-well scaling**.

***

## 5. Missing data structure and null handling

Goal: characterize exactly **how missingness occurs**, not just count NaNs.

### 5.1. Per-curve missingness

For each well and curve:

- Compute:
  - `% missing` (NaNs + null codes).
  - Number and length (in depth) of continuous missing segments.

Visualizations:

- Heatmap (well vs curve) of missing percentage.
- Depth-location missing matrix for a few representative wells (depth vs curve with color indicating null/valid).

This tells you:

- Which curves are usable as features or targets for each well.
- Whether missingness is at the **top, bottom, or internal segments**.
- How to design **gap-filling tasks** (e.g. mid-interval missing vs top/bottom truncation).[11][8]

### 5.2. Co-missingness patterns

- Analyze where multiple curves are missing simultaneously (e.g. GR and RT both missing over same depth).
- This matters for multi-log ML: if the target is missing where inputs are also missing, you cannot train on that region.

Outcome: define **imputation strategy**:

- Pure interpolation for short gaps.
- ML-based gap-filling for longer segments.
- Exclusion for wells/curves with too much missingness.

***

## 6. Depth-wise relationships and cross-plots

Goal: discover **petrophysical relationships** and non-linear correlations across curves.

### 6.1. Cross-plots per curve pair

For wells with multiple logs:

- Plot:
  - GR vs RHOB, GR vs NPHI, RHOB vs NPHI, RT vs NPHI, RT vs DT, etc.
- Optionally color by depth or by reservoir intervals if you know them.

Look for:

- Expected relationships:
  - High GR ↔ high shale fraction.
  - NPHI vs RHOB trends (gas effect, lithology lanes).
  - RT vs NPHI correlation in hydrocarbon zones.[12][6][13]
- Abnormal wells where these relationships break (measurement issues, bad calibration).

Outcome: decide:

- Which features have **meaningful predictive power** for your target.
- Whether to apply non-linear transforms (e.g. log10(RT), porosity transforms).

***

## 7. Log shape and continuity analysis

Goal: understand **curve shapes** as time-series along depth, to tune model types.

### 7.1. Visual log plots (per well)

For each key well:

- Plot along depth (y-axis depth, x-axis log value):
  - GR.
  - RT.
  - RHOB.
  - NPHI.
  - DT.

Assess:

- Smooth vs noisy segments.
- Abrupt spikes suggesting tool jumps or bad points.
- Depth intervals with extreme noise or constant values.

From this you define:

- Whether to **apply smoothing filters** (e.g. moving average, Savitzky-Golay) for specific curves.
- Where to apply **outlier removal** filters (e.g. median filters, robust z-score thresholds).

### 7.2. Autocorrelation and correlation length

For each curve (e.g. GR) in representative wells:

- Compute autocorrelation vs depth lag.
- Estimate correlation length (depth interval at which correlation drops to, say, 0.2–0.3).

This informs:

- Sequence window length for LSTM / Conv1D models:
  - If GR correlation length ~20 ft and your sampling step is 0.5 ft, a window of 40–60 samples is reasonable.[14][9][15]

***

## 8. Outlier/anomaly detection (univariate and multivariate)

Goal: identify **bad measurements** that should be masked before ML.

### 8.1. Univariate outliers

For each curve:

- Use robust metrics:
  - Median ± k * MAD (Median Absolute Deviation).
  - Quantile-based: values below Q1–1.5·IQR or above Q3+1.5·IQR.
- Flag and inspect outliers:
  - Are they isolated spikes?
  - Or entire intervals with unrealistic ranges?

Policy:

- For isolated spikes: set to NaN and later interpolate.
- For entire intervals: use ML-based gap-filling only if physically plausible; otherwise consider excluding.[10][8]

### 8.2. Multivariate outliers

On wells with multiple logs:

- For each depth point with full set of logs, fit:
  - Isolation Forest, or
  - Local Outlier Factor (LOF).

Visualize outliers in cross-plots:

- Are they systematic (e.g. curve shifted) or random?

This step supports robust ML by **neurologically cleaning pathological combinations of logs**.

***

## 9. Stationarity and scaling strategy

Goal: decide **how to scale and normalize** logs for ML.

### 9.1. Check stationarity per curve

- Plot running mean and variance of each curve along depth.
- Test for large shifts between formations (e.g. top vs base Eagle Ford).

If strong non-stationarity:

- Consider:
  - **Per-well** scaling (z-score within a well).
  - Or **per-formation** scaling if formation tops are known.

### 9.2. Decide on final scaling

Based on previous EDA:

- Choose for each curve:
  - StandardScaler (z-score) or MinMaxScaler.[16]
  - Global scaler (fitted on all wells) vs well-wise scaler.

Keep in mind:

- Many ML works on logs found **standardization** crucial for neural nets, but tree-based models (RF, XGBoost) can work on unscaled or lightly transformed data (just log10 on RT etc.).[5][11]

***

## 10. Model-target feasibility analysis

Goal: match **available curves** and their quality with your thesis ML tasks.

From all previous steps, determine:

- For each potential target (e.g. DT, RHOB, RT_log, or GR itself):
  - Number of wells with enough depth coverage and acceptable quality.
  - Availability of predictor curves.

Then define:

- **Primary ML experiment**:
  - With the largest, cleanest dataset (probably GR-only (gap-filling / denoising) or GR-based lithology classification if labels exist).[6][1][12]
- **Secondary ML experiment**:
  - Use the small subset of wells with multiple high-quality logs to do classic “missing log prediction” (e.g. predict density from GR+NPHI+RT).

This ensures the ML problems are well-posed for *your actual data*, not generic textbook cases.

***

## How this EDA tunes your preprocessing pipeline

Once an agent executes this EDA, they will have concrete answers to:

- **Curve selection**: which logs are usable, common, and predictive.
- **Curve mapping**: how to harmonize mnemonics across wells.
- **Depth resampling**: whether to resample to a common grid and at what step.
- **Outlier and noise handling**: where to clip, null, smooth, or exclude.
- **Missing data strategy**: which gaps to interpolate vs which to fill with ML vs which to drop.
- **Scaling strategy**: per-well vs global, z-score vs MinMax, transforms like log10(RT).
- **Sequence design**: correlation lengths → window sizes for LSTM/Conv1D.
- **Task definition**: which curves can realistically serve as targets and for which subset of wells.

If you like, the next step can be: take this EDA plan and translate it into a **single, modular Python notebook structure** (with function stubs) that your “expert agent” can fill in and execute on your Dimmit LAS folder.

[1](https://towardsdatascience.com/exploratory-data-analysis-with-well-log-data-98ad084c4e7/)
[2](https://pypi.org/project/lasio/0.7.5/)
[3](https://lasio.readthedocs.io/en/v0.30/)
[4](https://towardsdatascience.com/loading-multiple-well-log-las-files-using-python-39ac35de99dd/)
[5](http://arxiv.org/pdf/2010.04186.pdf)
[6](https://www.sciencedirect.com/science/article/abs/pii/S0926985122000763)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10560031/)
[8](https://www.e3s-conferences.org/articles/e3sconf/pdf/2022/19/e3sconf_esat2022_01019.pdf)
[9](https://arxiv.org/html/2412.00718v1)
[10](https://rokdoc.ikonscience.com/exploratory-data-analysis-eda-tool-joins-the-deepqi-suite)
[11](https://www.sciencedirect.com/science/article/pii/S0926985123000708)
[12](https://publications.eai.eu/index.php/IoT/article/view/5634)
[13](https://www.mdpi.com/1424-8220/25/3/836)
[14](http://arxiv.org/pdf/2307.10253.pdf)
[15](https://www.nature.com/articles/s41598-025-95709-0)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_8127a2be-04d3-43c8-8149-ebff394f60bd/8bc5d414-ceef-4ed8-84f7-55b7d240897c/2023_Sonic_Log_Prediction_Volve_Field-1.pdf)
[17](https://riojournal.com/article/23676/download/pdf/)
[18](https://arxiv.org/pdf/2210.05597.pdf)
[19](https://www.mdpi.com/2313-433X/9/7/136/pdf?version=1688693413)
[20](https://www.mdpi.com/2227-9717/11/12/3421/pdf?version=1702461863)
[21](https://www.mdpi.com/2076-3417/15/6/3020)
[22](https://community.databricks.com/t5/technical-blog/processing-las-well-log-files-and-other-semi-structured-data/ba-p/133583)
[23](https://www.linkedin.com/posts/andymcdonaldgeo_loading-and-exploring-well-log-las-files-activity-6949693809307877376-uYWJ)
[24](https://www.kaggle.com/code/giangpt/exploratory-data-analysis-with-well-logs)
[25](https://www.youtube.com/watch?v=GwAbfriuHr4)
[26](https://www.youtube.com/watch?v=cy9Jrm6u07w)
[27](https://fxis.ai/edu/getting-started-with-lasio-reading-and-writing-log-ascii-standard-files-in-python/)
[28](https://ieeexplore.ieee.org/document/10677822/)
[29](https://github.com/andymcdgeo/Petrophysics-Python-Series)
[30](https://lasio.readthedocs.io)
[31](https://www.youtube.com/watch?v=8U4gxMJybJs)