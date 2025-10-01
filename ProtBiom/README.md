# ProtBiom

Protein-based biomarker scores in R.
This package helps you compute two published protein scores—**ProtFI** and **ProtMort**—or your **own custom score** from a list of proteins and coefficients. It handles **case-insensitive** column matching, optional **z-scaling**, and **fallbacks** for missing proteins.

---

## Installation

```r
# From GitHub
# install.packages('devtools')
devtools::install_github("swiergarst/ProtFI", subdir = "ProtBiom")
```

---

## Quick start

```r
library(ProtBiom)

# Example input: one row per participant, columns are proteins (and age)
head(df)  # your protein data.frame

# 1) Frailty score (ProtFI)
fi <- predictProtFI(df, id_col = "SampleID", scale = TRUE)

# 2) Mortality score (ProtMort)
mort <- predictProtMort(df, id_col = "SampleID", scale = TRUE)

# 3) Custom score (provide your own coefficients)
my_coefs <- data.frame(
  Protein     = c("AGE", "ProteinA", "ProteinB"),
  Coefficient = c(0.12, 0.34, -0.08)
)
custom <- protpredict(
  data    = df,
  biomarker  = "MyProtScore",   # output column name
  id_col     = "SampleID",
  scale      = TRUE,            # or FALSE if already z-scored
  coef_df    = my_coefs         # required for custom scores
)
```

---

## Important assumptions

* **Imputation required:**
  This package expects **already imputed data**. The original ProtFI and ProtMort papers used **KNN imputation with k = 5**.
  **Any participant with `NA` in any required predictor (after fallbacks) will receive `NA` for the score.**
  The function does not impute; warnings about missing values are **serious**—**impute your data before using this package**.

* **Z-transformation required:**
  Predictors must be **z-standardized (mean ~0, SD ~1)**.

  * If they are not, set `scale = TRUE` and the function will z-scale them for you.
  * If you use `scale = FALSE` and your inputs are not correctly z-standardized, **your results are not valid**—do **not** ignore this warning unless the input data is a subset of a already Z-transformed dataset.

* **Fallbacks for missing proteins:**
  For ProtFI and ProtMort, if a required protein is missing, it is replaced by its **most strongly correlated protein from the UK Biobank Cardiometabolic panel** (defined in the bundled `highcor_tbl`).

  * If you want a different replacement scheme, or correlations from another panel, supply your own `highcor_df`.

---

## Main functions

### `protpredict()`

Compute a protein-based score.

**Arguments**

* `data` (data.frame): rows = participants, columns = proteins (plus `age` if needed).
* `biomarker` (character): `"ProtFI"`, `"ProtMort"`, or a custom label.

  * `"ProtFI"` / `"ProtMort"` auto-load their package coefficients.
  * Any other label requires `coef_df`.
* `id_col` (optional): input ID column name (case-insensitive). If `NULL`, tries `"sampleid"`. If not found, rownames or row numbers are used.
* `id_name` (character): output ID column name (default `"sampleid"`).
* `scale` (logical): if `TRUE`, z-scale predictors with mean/SD from `data`. If `FALSE`, predictors must already be z-standardized.
* `coef_df` (optional): data.frame with columns `"Protein"` and `"Coefficient"`. **Not allowed** when `biomarker` is `"ProtFI"` or `"ProtMort"`.
* `highcor_df` (optional): fallback map for missing proteins (defaults to `highcor_tbl`).

**Return**

A data.frame with the (optional) ID column and one score column named exactly as in `biomarker`.

### Shortcuts

* `predictProtFI(data, id_col = NULL, scale = FALSE, ...)`
* `predictProtMort(data, id_col = NULL, scale = FALSE, ...)`

---

## How matching and fallbacks work

* **Case-insensitive matching:** column names are matched ignoring capitalization and extra spaces.
  In documentation and examples, we keep capitalization consistent for readability.
* **Coefficients source:**

  * `ProtFI` → package dataset `ProtFIcoef_tbl`
  * `ProtMort` → package dataset `ProtMortcoef_tbl`
  * Custom label → you must pass `coef_df`.
* **Missing predictors:**

  * If a required protein is missing, it is replaced by its **strongest correlate** from `highcor_tbl` (or your `highcor_df`).
  * **Exception:** `age` may **not** be replaced. If coefficients include `age` but `age` is missing in `data`, the function stops with an error.
* The function reports replacements with messages like
  `Replaced: 'ProteinX' -> 'ProteinY'.`

---

## Scaling and NA handling

* If `scale = TRUE`:

  * Predictors are z-scaled using **mean and SD from your `data`**.
  * If there are NAs, the function **continues**, but those entries **remain `NA`** and participants with missing predictors will have **`NA` scores**.
  * The function only warns; it does not impute.
  * Scaling fails if any predictor has SD = 0 or SD is `NA`.

* If `scale = FALSE`:

  * Your inputs must already be z-standardized.
  * If not, the function will give a **serious warning**. Results will not be trustworthy if you ignore it.
  * **NA values propagate** to the score (no internal imputation).

> **Row-level behavior:** The score is a linear combination of predictors. If **any** used predictor is `NA` for a participant (after fallbacks and optional scaling), that participant’s score will be `NA`.

---

## Required package data

Bundled datasets (loaded automatically):

* `ProtFIcoef_tbl` — coefficients for ProtFI
* `ProtMortcoef_tbl` — coefficients for ProtMort
* `highcor_tbl` — fallback map: `"Protein"` → `"Strongest correlated protein"`

You can override any of these with your own `coef_df` or `highcor_df`.

---

## Output columns

* `sampleid` (or your `id_name`)
* One score column named exactly as you set in `biomarker`
  (e.g., `"ProtFI"`, `"ProtMort"`, or `"MyProtScore"`)

---

## Common errors and messages

* **`Column 'X' not found (case-insensitive).`**
  The required protein is not in `data`.
* **`Required column 'age' is missing in data.`**
  `age` appears in the coefficients but not in your input. Add it; it cannot be replaced.
* **`No replacement found in highcor_tbl for missing variable: X`**
  Add a row for `X` to your `highcor_df`, or include `X` in your data.
* **`Custom coef_df is not allowed when biomarker = 'ProtFI'/'ProtMort'.`**
  For the built-in scores, coefficients are fixed.
* **Standardization warning when `scale = FALSE`:**
  Your inputs look unscaled. Either set `scale = TRUE` or standardize upstream. Ignoring this will invalidate results.
* **Scaling error:**
  SD is 0 or NA for a predictor → remove constant columns or impute first.

---

## Citation

If you use **ProtFI** or **ProtMort**, please cite the original paper; doi: https://doi.org/10.1101/2025.09.19.25336152
If you use **ProtBiom**, please cite this repository as well.
