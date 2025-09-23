#' Coefficients for ProtFI
#'
#' @format A data frame with columns:
#' \describe{
#'   \item{Protein}{character}
#'   \item{Coefficient}{numeric}
#' }
#' @source Derived for this study (also available in https://github.com/swiergarst/ProtFI/blob/master/output_linear/coefs_frail/selected_models/combine_coefs_frailty_cmb_ffs.csv)
"ProtFIcoef_tbl"

#' Coefficients for ProtMort
#'
#' @format A data frame with columns:
#' \describe{
#'   \item{Protein}{character}
#'   \item{Coefficient}{numeric}
#' }
#' @source Derived for this study (also available in https://github.com/swiergarst/ProtFI/blob/master/output_linear/coefs_mort/selected_models/combine_coefs_mort_cmb_ffs.csv)
"ProtMortcoef_tbl"

#' Strongest correlations between Cardiometabolic proteins in the UK Biobank used for fallback replacement in case of missingness
#'
#' @format A data frame with columns:
#' \describe{
#'   \item{Protein}{character}
#'   \item{Strongest correlated protein}{character}
#'   \item{Correlation coefficient}{numeric}
#' }
#' @source Derived from UK Biobank analyses (also available in Supplementary Table 8)
"highcor_tbl"
