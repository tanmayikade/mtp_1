# WELL LOG PREDICTION THESIS: PROJECT CHARTER & CONTEXT DOCUMENT
## Goal -  Context -  Dataset -  Strategy

***

## EXECUTIVE FRAMING

You are a **final-year Masters student at IIT Kharagpur** preparing a thesis on **well log data prediction using machine learning in the Eagle Ford Formation, Dimmit County, Texas**. This document establishes the **project charter, realistic scope, and data-driven strategy** based on actual RRC well log inventory you have inspected.

**Critical Shift from Synthetic to Real Data:**
- Previous guidance assumed synthetic data for speed.
- **You now have access to real LAS files** (6022 total records, ~500 oil wells with GR curves).
- This doc reframes the thesis scope around **your actual data**, not generic templates.

***

## 1. PROJECT GOAL & THESIS STATEMENT

### 1.1 Primary Research Question

> **"Can machine learning models efficiently predict missing or degraded well log curves in the Eagle Ford Formation using available petrophysical measurements, and which ML approach offers the best balance of accuracy, interpretability, and operational deployment?"**

### 1.2 Specific Objectives

1. **Objective 1 (Main):** Develop and compare ML/DL models for **GR log gap-filling and denoising** across 30–50 oil wells in Dimmit County Eagle Ford, achieving **R² > 0.95** with interpretable feature importance.

2. **Objective 2 (Secondary):** On the small subset of multi-log wells (GR + density or resistivity), demonstrate **cross-log prediction** (e.g., GR → RHOB or RT) as a proof-of-concept for richer log synthesis.

3. **Objective 3 (Practical):** Recommend a **production-ready model** (XGBoost or LSTM) with deployment guidelines, operational constraints, and cost-benefit analysis vs. conventional log re-running.

### 1.3 Thesis Contribution

Your thesis will be the **first comprehensive ML study on well log prediction for oil wells in Dimmit County Eagle Ford**, filling a gap between:
- Generic synthetic studies (no field validation).
- Multi-log studies from other basins (Eagle Ford geology-specific tuning lacking).

***

## 2. CONTEXT & LANDSCAPE

### 2.1 Eagle Ford Formation in Dimmit County

**Geological Setting:**
- Location: South Texas, Dimmit County (1,334 sq miles).
- Depth: 4,000–12,000 ft TVD.
- Thickness: 300–500 ft.
- Depositional environment: Turonian–Cenomanian unconventional shale.

**Production Profile (Dimmit County):**
- Current production: ~117,500 barrels/day (2nd highest in Eagle Ford basin).
- Petroleum windows:
  - **Oil** (North): Your focus.
  - Wet gas/condensate (Central).
  - Dry gas (South).
- Primary operators: Anadarko, Chesapeake, Newfield.

**Why Eagle Ford is Ideal for Your Study:**
- Mature, well-logged field with extensive RRC database.
- Oil-window wells have consistent petrophysical behavior (vs. gas/condensate).
- High commercial interest → your results have real impact.
- Enough wells available to build statistically robust ML models.

### 2.2 Current State of Well Log Availability (RRC Search Results)

**Raw Inventory:**
- Total LAS records in Dimmit County: **1,330** (filtered from 6022 multi-type files).
- Oil wells (O) with LAS: ~**500**.
- Dominant curve types:
  - **GR only**: ~475 wells.
  - **GR + Resistivity**: 1 well.
  - **GR + Density**: 2 wells.
  - **GR + Induction**: 2 wells.
  - Other combinations: <5 wells.

**Data Maturity:**
- All files are public (Texas RRC open data, no access restrictions).
- LAS format is standard (easily parsed with `lasio` library).
- No known data use restrictions for academic thesis.

### 2.3 Why the ML Problem Matters

**Operational Problem:**
- Well logging is expensive: sonic logs cost $2–5K per well, sonic re-runs $5–10K.
- Wireline tools fail, boreholes wash out → log gaps are common.
- Operators need rapid, cost-effective ways to fill missing log segments for:
  - Petrophysical property calculation (porosity, permeability).
  - Seismic tie optimization (synthetic seismograms need sonic).
  - Completion design inputs.

**Current Industry Solution:**
- Empirical correlations (poor generalization across wells/formations).
- Physicist-based rock models (require well-calibration, slow).
- Re-logging (expensive, slow, sometimes unsafe).

**ML Opportunity:**
- Data-driven models can learn well-specific and formation-specific patterns.
- Faster, cheaper, can be deployed operationally.

***

## 3. YOUR DATASET: REAL INVENTORY & REALISTIC SCOPE

### 3.1 What You Actually Have

From your RRC search:

| Metric | Count |
|--------|-------|
| **Total LAS records, Dimmit County** | 1,330 |
| **Oil wells (O-code) with LAS** | ~500 |
| **GR-only oil wells** | ~475 |
| **Multi-log oil wells (GR+RT/DEN/other)** | <5 |
| **Your downloaded GR samples** | 30 |

### 3.2 Collection Target for 3-Day Thesis

**For Primary ML Experiment (GR Gap-Filling):**
- Target: **30–50 oil wells** with GR LAS.
  - Download method: Manual RRC search + batch download.
  - Rationale: ~40 wells × ~3,000 samples/well ≈ **120,000 training rows** (sufficient for tree-based and 1 LSTM variant).
  - Time to collect: **2–4 hours** (download + organize).

**For Secondary Experiment (Multi-Log Prediction):**
- Use all 3–5 multi-log wells available.
- Task: GR → Density or RT (small N, but demonstrates cross-curve capability).

**Sample Organization:**
```
Dimmit_EagleFord_LAS/
├── raw_las/
│   ├── API_1_GR.las
│   ├── API_2_GR.las
│   └── ... (40+ files)
├── metadata.csv          # API, operator, depth range, etc.
└── processed/
    ├── master_welllog.csv
    ├── cleaned_data.pkl
    └── ...
```

### 3.3 Expected Data Characteristics (from EDA insights)

**GR Curves:**
- Range: typically x-y API.
- Sampling: ~x ft (variable per well).
- Depth extent: x–y ft (varies by well).
- Missingness: x-y% nulls, mainly at top/bottom, occasional mid-interval gaps.
- Quality: Most wells are serviceable; a few will have anomalies (tool drift, bad calibration).

**Implications for ML:**
- Need per-well and global outlier detection.
- Depth resampling to common grid (e.g., 1 ft) ensures uniform sequence modeling.
- Window-based sequence design (~20–50 ft windows for LSTM).

***

## 4. MACHINE LEARNING STRATEGY (BASED ON YOUR DATA)

### 4.1 Task Definition: Primary Experiment

**Task Name:** GR Log Gap-Filling via Sequence-to-One Regression

**Input:**
- Sliding window of GR values along depth (e.g., ±10 samples around a center point).
- Optionally: depth coordinate (normalized), well ID (embeddings).

**Output:**
- GR value at center of window (regression target).

**Why This Task?**
1. **Realistic:** Operators frequently have missing GR intervals.
2. **Data-abundant:** You have ~475 GR-only wells, each contributing many sequence examples.
3. **Model-agnostic:** Can use RF, XGBoost, LSTM, Conv1D, all suitable.
4. **Interpretable:** Learned feature importance reveals depth/window effects.

**Expected Difficulty:** **Moderate**
- GR is a well-understood log; strong local correlations exist.
- Expect R² > 0.95 for gap-filling within same well.
- Cross-well generalization: R² > 0.90 (harder, but achievable).

### 4.2 Task Definition: Secondary Experiment

**Task Name:** Cross-Curve Log Prediction (Case Study)

**Input:**
- GR + other available logs (RT, RHOB, NPHI, PEF) if present.

**Output:**
- Missing or target log (e.g., density, sonic).

**Scope:**
- Limited to 3–5 wells with multiple logs.
- Frame as **illustrative proof-of-concept**, not main statistical result.
- Honest caveat in thesis: "N wells = 3 is too small for robust generalization; this case study demonstrates conceptual feasibility."

**Why Include This?**
- Shows you understand classic "predict missing log from other logs" problem.
- Demonstrates ML scalability: same pipeline works whether input is 1 log or 6.

***

## 5. ML MODEL SELECTION

### 5.1 Why These Three Models?

Given your 3-day deadline and data characteristics:

| Model | Why | Local/Colab | Expected R² |
|-------|-----|-------------|------------|
| **Random Forest** | Baseline, fast, interpretable, handles raw data | **Local** | 0.93–0.95 |
| **XGBoost** | SOTA gradient boosting, faster inference, feature importance | **Local** | 0.95–0.97 |
| **LSTM (Conv1D)** | Sequence modeling, deep learning showcase, GPU-friendly | **Colab** | 0.97–0.99 |

### 5.2 Realistic Performance on Real Eagle Ford Data

**Based on literature and your data characteristics:**

- **RF:** R² ~0.93 (robust baseline, struggles with depth-dependent nonlinearity).
- **XGBoost:** R² ~0.96 (captures nonlinear GR patterns, fast training).
- **LSTM/Conv1D:** R² ~0.98 (if trained on sufficient sequences, GPU-enabled).

**Why Not Higher?**
- Real logs have noise, measurement error, formation heterogeneity.
- GR-only models lack lithology labels → predicting raw curve shape (harder than facies).
- Cross-well generalization (test on unseen wells) is harder than within-well.

***

## 6. PREPROCESSING PIPELINE TUNING STRATEGY

### 6.1 Based on EDA, Your Preprocessing Will Address

**1. Depth Standardization**
- Resample each well to common grid (e.g., 1 ft intervals).
- Keep track of original depth for interpretability.

**2. Outlier & Noise Handling**
- Flag GR spikes (univariate: median ± 3·MAD).
- Median-filter noisy segments (window ~5–10 ft).
- Hard-clip unrealistic GR (e.g., <0 or >250 API).

**3. Missing Value Strategy**
- Short gaps (<10 ft): Linear interpolation.
- Long gaps (>100 ft): Mark as "do not train" (realistic gap-filling scenario).
- Top/bottom truncation: Accept as-is (common in practice).

**4. Scaling & Normalization**
- **For tree models (RF, XGBoost):** Minimal processing; log10(RT) only if included.
- **For LSTM:** StandardScaler per well (z-score normalization).
- **Rationale:** Neural nets sensitive to scale; trees scale-invariant.

**5. Train/Val/Test Split**
- **By well** (not random rows):
  - Train: 70% of wells (~28 wells).
  - Validation: 15% of wells (~6 wells).
  - Test: 15% of wells (~6 wells).
  - Reason: Avoids data leakage; tests true generalization.

***

## 7. CONCRETE WORKFLOW FOR YOUR 3 DAYS

### Day 1: Data Collection & EDA

**Morning (4–6 hours):**
1. Download 40–50 GR LAS files from RRC (batch + organize).
2. Parse all LAS with `lasio` into master CSV.
3. Run full EDA notebook:
   - Curve inventory, depth ranges, missingness patterns.
   - Per-curve statistics (min, max, mean, std).
   - Outlier flagging heatmaps.
   - Cross-well distribution comparison.

**Output:** `dimmit_gr_master_eda.csv`, `eda_report.html`, 8–10 diagnostic plots.

### Day 2: Preprocessing & Model Training

**Morning (6–8 hours):**
1. Implement preprocessing pipeline (depth resampling, outlier removal, scaling).
2. Create training sequences (window-based for LSTM; row-based for tree models).
3. Train locally:
   - Random Forest (30 min).
   - XGBoost (1 hour, with early stopping).

**Afternoon–Night (6–10 hours, parallel):**
4. Upload cleaned data to Colab.
5. Train LSTM/Conv1D on GPU (2–4 hours overhead; runs while you code).
6. Download trained models.

**Output:** 3 trained models, predictions on test set, performance metrics.

### Day 3: Analysis & Report

**Morning–Afternoon (8–10 hours):**
1. Compare all 3 models: R², RMSE, MAE, inference time.
2. Feature importance analysis (RF, XGBoost SHAP values).
3. Generate 8–10 visualizations.
4. Write report (15–20 pages):
   - Intro + literature review (RRC data, Eagle Ford geology).
   - Data & EDA findings.
   - Preprocessing rationale.
   - Results & model comparison.
   - Deployment recommendations.
   - Conclusion & future work.

**Output:** Final thesis draft, all code, all visualizations, submission-ready package.

***

## 8. REALISTIC SUCCESS METRICS

### 8.1 Minimum Viable Thesis (Will Pass)

- ✓ Downloaded 15–20 real GR LAS wells from Dimmit County.
- ✓ Cleaned and preprocessed data properly (outliers removed, normalized).
- ✓ Trained Random Forest + XGBoost locally.
- ✓ Achieved R² > 0.93 on test set.
- ✓ 12–15 page report with results, visualizations, discussion.
- ✓ Code reproducible (requirements.txt, clear comments).

**Grade Expectation:** B+ to A–

### 8.2 Good Thesis (Will Impress)

- ✓ All above + 30–40 real oil wells (Dimmit County, Eagle Ford).
- ✓ Comprehensive EDA (10+ diagnostic plots, detailed missing-data analysis).
- ✓ Thoughtful preprocessing with well-specific scaling.
- ✓ All 3 models (RF, XGBoost, LSTM) working, compared fairly.
- ✓ R² > 0.95 (XGBoost), R² > 0.97 (LSTM).
- ✓ SHAP/feature importance with geological interpretation.
- ✓ 16–18 page report, professionally written.
- ✓ Deployment recommendations with E&P context.

**Grade Expectation:** A to A+

### 8.3 Excellent Thesis (Will Wow)

- ✓ All above + well-stratified data (different petroleum windows).
- ✓ Cross-validation by well or k-fold analysis.
- ✓ Secondary experiment: GR → RHOB/RT on multi-log wells (case study).
- ✓ Uncertainty quantification (prediction intervals, Monte Carlo).
- ✓ Domain-specific insights (Eagle Ford stratal trends, operator-specific patterns).
- ✓ LSTM attention weights visualized → interpretable deep learning.
- ✓ 18–20 page report, publication-ready.
- ✓ Operational cost-benefit analysis (when to use ML vs. re-log).

**Grade Expectation:** A+ (potential for publication)

***

## 9. YOUR UNIQUE POSITIONING

### 9.1 What Makes This Thesis Novel

1. **First ML study on Dimmit County Eagle Ford oil wells** (most prior work is synthetic or other basins).
2. **Real RRC data** (not simulated → validates on actual operational data).
3. **GR-centric approach** (acknowledges data reality; doesn't force multi-log models on limited wells).
4. **Production-ready focus** (recommends specific model for deployment).

### 9.2 Anticipated Contribution to Field

- Demonstrates ML can reduce well log re-running costs in Eagle Ford.
- Provides Dimmit County operators a reproducible workflow for log gap-filling.
- Bridges academic ML methods and E&P operational constraints.

***

## 10. RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Slow RRC download** | Lose 4–6 hours to data collection | Start immediately; download in parallel while coding EDA |
| **LAS parsing errors** (corrupted files) | 10–20% of wells unusable | Build robust error-handling in lasio loop; drop bad wells early |
| **Low model performance** (R² < 0.90) | Thesis premise weakened | Fallback to GR denoising (easier task); or synthetic blend if needed |
| **LSTM training fails on Colab** | Lose 2–3 hours | Use Kaggle GPU as backup; or simplify to Conv1D (faster) |
| **Report writing overruns** | Incomplete submission | Pre-write sections during data/model runs; use templates |

***

## 11. TOOLS & ENVIRONMENT

**Languages & Libraries:**
- Python 3.9+
- `lasio` (LAS parsing)
- `pandas`, `numpy` (data handling)
- `scikit-learn` (RF, preprocessing)
- `xgboost` (gradient boosting)
- `tensorflow`/`keras` (LSTM, Colab)
- `matplotlib`, `seaborn` (visualization)
- `shap` (explainability)

**Compute:**
- Mac Air M2: RF, XGBoost (local).
- Google Colab (free GPU): LSTM training.
- RRC database (public, free): well log data.

**Storage:**
- Raw LAS: ~100–200 MB (40 wells).
- Processed data: ~50 MB.
- Models + results: ~50 MB.
- **Total project size: <500 MB** (easily manageable).

***

## 12. THESIS STRUCTURE (16–20 pages)

1. **Cover Page & Abstract** (1 page)
2. **Introduction** (2 pages): Problem, Eagle Ford context, thesis objectives.
3. **Literature Review** (2 pages): ML for well logs, Eagle Ford geology, Dimmit County specifics.
4. **Data & Methodology** (4 pages): RRC inventory, EDA findings, preprocessing pipeline, ML models.
5. **Results** (3 pages): Performance metrics, visualizations, feature importance.
6. **Discussion** (3 pages): Model comparison, geological interpretation, deployment feasibility.
7. **Conclusions & Future Work** (1 page)
8. **References** (1 page): 20–30 citations.

**Visualizations (8–10):**
- EDA: depth range, missing % heatmap, curve distributions.
- Results: predicted vs actual scatter (per model), residual plots, feature importance bar charts.
- Comparison: R² vs RMSE vs inference time (bubble chart or table).

***

## 13. KEY DECISIONS YOU'VE MADE (DOCUMENTED)

✅ **Use REAL RRC data** (not synthetic) → More credible, higher impact.  
✅ **Focus on GR logs** (only ~475 oil wells, but sufficient) → Statistically robust.  
✅ **Gap-filling as primary task** (vs. lithology classification) → Operationally realistic.  
✅ **3-model comparison** (RF, XGBoost, LSTM) → Shows breadth of ML approaches.  
✅ **Deploy locally + Colab** (no local GPU needed) → Practical given Mac M2.  
✅ **16–20 page report** (not 50-page theoretical thesis) → Concise, professional.

***

## 14. FINAL CHECKPOINT: ARE YOU READY?

**Before you begin downloading data, confirm:**

- [ ] You have 30–50 GR LAS files target from Dimmit County (RRC accessible).
- [ ] You understand the task: gap-fill GR curves via sequence models.
- [ ] You have Python + required libraries ready (or Colab account + Drive setup).
- [ ] You have pre-processing EDA doc (from earlier guidance).
- [ ] Your guide approved the scope (real data, 3 models, 16–20 page thesis).

**If all YES → Proceed immediately to download + EDA.**  
**If any NO → Clarify with your guide NOW, don't waste time guessing.**

***

## SUMMARY: THESIS CHARTER

| Aspect | Your Project |
|--------|--------------|
| **Title** | Machine Learning-Based Well Log Prediction in Eagle Ford Formation: A Dimmit County Case Study |
| **Data** | 30–50 real GR LAS oil wells from RRC (Dimmit County, Eagle Ford). |
| **Task** | GR gap-filling + optional cross-log prediction (secondary). |
| **Models** | Random Forest, XGBoost, LSTM (3 approaches, fair comparison). |
| **Success Metric** | R² > 0.95 (XGBoost); operational deployment recommendation. |
| **Timeline** | 3 days (real data collection, EDA, model training, report). |
| **Deliverables** | Code, 30–50 processed LAS wells, 3 trained models, 16–20 page thesis. |
| **Impact** | First comprehensive ML study on Dimmit County Eagle Ford oil wells; practical E&P value. |

***

**You now have a data-driven, realistic thesis charter grounded in actual RRC inventory. Execute this plan with confidence.**

**Next step:** Download your 40–50 GR LAS files. Begin EDA. Report progress.**