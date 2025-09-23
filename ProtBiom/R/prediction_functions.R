#' Protein-based score prediction (case-insensitive matching)
#'
#' Rules:
#' - Coefficients are chosen by `biomarker` or taken from `coef_df`.
#' - Missing predictors are replaced by their strongest correlate from `highcor_tbl`,
#'   except 'age': if 'age' is missing in `data` (and present in `coef_df`), error.
#' - If `scale = TRUE` and any predictor contains NA, error (expects imputed data; original paper used KNN with k = 5).
#' - If `scale = FALSE`, predictors must already be z-standardized (~ mean 0, sd 1); otherwise error advising `scale = TRUE`.
#' - With `scale = FALSE`, NA values propagate to the score (no internal imputation).
#' - Matching of column names is case-insensitive.
#'
#' @param data A data.frame with participants (rows).
#' @param biomarker Character. "ProtFI", "ProtMort", or a custom label.
#'        If "ProtFI"/"ProtMort", coefficients are auto-loaded from package data
#'        (`ProtFIcoef_tbl` / `ProtMortcoef_tbl`) and custom `coef_df` is not allowed.
#'        For any other value, you must provide `coef_df`.
#' @param id_col Optional input ID column name in `data` (case-insensitive). If NULL, the function
#'        will try to auto-detect a column named "sampleid" (case-insensitive). If still not found,
#'        no ID column is returned.
#' @param id_name Output ID column name (default "sampleid").
#' @param scale Logical; if TRUE, z-scale used predictors with mean/SD from `data`.
#' @param coef_df Optional override for coefficients (columns "Protein","Coefficient").
#' @param highcor_df Optional override for fallback map (defaults to package data `highcor_tbl`,
#'        columns "Protein","Strongest correlated protein").
#' @return A data.frame with the ID column (if available, named by `id_name`) and a score column
#'         named exactly as provided in `biomarker`.
#' @export
protpredict <- function(data,
                        biomarker  = c("ProtFI", "ProtMort", "custom")[1],
                        id_col     = NULL,
                        id_name    = "sampleid",
                        scale      = FALSE,
                        coef_df    = NULL,
                        highcor_df = NULL) {
  stopifnot(is.data.frame(data))
  score_col <- as.character(biomarker)  # preserve given name for output

  # Data loader
  .get_pkgdata <- function(obj) {
    if (exists(obj, inherits = TRUE)) return(get(obj, inherits = TRUE))
    pkg <- tryCatch(utils::packageName(), error = function(e) NULL)
    if (is.null(pkg) || is.na(pkg)) {
      if (requireNamespace("pkgload", quietly = TRUE)) {
        pkg <- tryCatch(pkgload::pkg_name(), error = function(e) NULL)
      }
    }
    if (!is.null(pkg)) {
      try(suppressWarnings(utils::data(list = obj, package = pkg, envir = environment())), silent = TRUE)
      if (exists(obj, envir = environment(), inherits = FALSE)) return(get(obj, envir = environment()))
    }
    try(suppressWarnings(utils::data(list = obj, envir = environment())), silent = TRUE)
    if (exists(obj, envir = environment(), inherits = FALSE)) return(get(obj, envir = environment()))
    p <- file.path("data", paste0(obj, ".rda"))
    if (file.exists(p)) {
      load(p, envir = environment())
      if (exists(obj, envir = environment(), inherits = FALSE)) return(get(obj, envir = environment()))
    }
    stop("Could not load dataset '", obj, "'. Ensure data/", obj,
         ".rda exists and the object inside is named '", obj, "'.")
  }

  tolow    <- function(x) tolower(trimws(x))
  ci_match <- function(x, table) match(tolow(x), tolow(table))
  need_col <- function(df, name) {
    j <- ci_match(name, names(df))
    if (is.na(j)) stop("Column '", name, "' not found (case-insensitive).")
    names(df)[j]
  }

  # load coefficients based on biomarker
  key <- tolower(score_col)
  if (is.null(coef_df)) {
    if (key == "protfi") {
      coef_df <- .get_pkgdata("ProtFIcoef_tbl")
    } else if (key == "protmort") {
      coef_df <- .get_pkgdata("ProtMortcoef_tbl")
    } else {
      stop("For biomarker '", score_col, "' please provide `coef_df` with columns ",
           "'Protein' and 'Coefficient'.")
    }
  } else {
    # If ProtFI/ProtMort, do not allow input of own coef_df
    if (key %in% c("protfi", "protmort", "protfI")) {
      stop("Custom `coef_df` is not allowed when biomarker = '", score_col, "'. ",
           "Use biomarker='", score_col, "' without `coef_df`, or choose a custom biomarker label.")
    }
  }

  # Load highcor_df default
  if (is.null(highcor_df)) {
    highcor_df <- .get_pkgdata("highcor_tbl")
  }

  stopifnot(is.data.frame(coef_df), is.data.frame(highcor_df))

  # Set the required columns in coef_df and highcor_df
  pcol  <- need_col(coef_df,    "Protein")
  bcol  <- need_col(coef_df,    "Coefficient")
  hpcol <- need_col(highcor_df, "Protein")
  hbcol <- need_col(highcor_df, "Strongest correlated protein")

  req <- as.character(coef_df[[pcol]]) #All needed coefficients
  bet <- as.numeric(coef_df[[bcol]]) #The betas of the coefficients
  if (any(!is.finite(bet))) stop("Non-finite coefficients in `coef_df`.")
  names(bet) <- req

  dn   <- names(data)
  used <- character(length(req))

  # map each required variable to data (or replacement protein when not available)
  for (i in seq_along(req)) {
    v <- req[i]
    j <- ci_match(v, dn)
    if (!is.na(j)) {
      used[i] <- dn[j]
    } else {
      if (tolow(v) == "age") stop("Required column 'age' is missing in `data`.")
      k <- which(tolow(highcor_df[[hpcol]]) == tolow(v))[1]
      if (is.na(k)) stop("No replacement found in `highcor_tbl` for missing variable: ", v)
      repl <- as.character(highcor_df[[hbcol]][k])
      j2   <- ci_match(repl, dn)
      if (is.na(j2)) {
        stop("Column '", v, "' is missing and its replacement '", repl,
             "' is also not present in `data`.")
      }
      used[i] <- dn[j2]
      message("Replaced: '", v, "' -> '", repl, "'.")
    }
  }

  # Use original order and rename to original names
  X <- data[, used, drop = FALSE]
  names(X) <- req

  # Print error message when NAs are present in dataframe
  if (anyNA(X)) {
    message("This function expects imputed data. Original paper used KNN with 5 nearest neighbors. ",
         "Function now runs on unimputed data, this is not advised.")
  }

  # Check when scale=F whether predictors are already z-standardized
  if (!isTRUE(scale)) {
    tol_mean <- 0.1
    tol_sd   <- 0.1
    mu  <- vapply(X, function(x) mean(x, na.rm = TRUE), numeric(1))
    sdx <- vapply(X, function(x) stats::sd(x, na.rm = TRUE), numeric(1))
    bad <- (is.finite(sdx) & (abs(mu) > tol_mean | abs(sdx - 1) > tol_sd))
    if (any(bad, na.rm = TRUE)) {
      message("Predictors appear not standardized (z-scored). ",
           "Turn `scale = TRUE` to apply z-scaling on the fly, or provide pre-standardized data. ",
           "Please ONLY ignore this message when standardized the data on larger dataframe")
    }
  }

  # Z-scaling
  if (isTRUE(scale)) {
    if (anyNA(X)) {
      message("Please be aware that unimputed data is now used for scaling.")
    }
    mu  <- vapply(X, function(x) mean(x, na.rm = TRUE), numeric(1))
    sdx <- vapply(X, function(x) stats::sd(x,   na.rm = TRUE), numeric(1))
    bad <- !is.finite(sdx) | sdx == 0
    if (any(bad)) {
      stop("Cannot scale; SD is NA or 0 for: ",
           paste(names(sdx)[bad], collapse = ", "))
    }
    X <- as.data.frame(scale(X, center = mu, scale = sdx), check.names = FALSE)
  }

  # Calculate biomarker score. This will output NA if a row has NA in any of the required proteins
  score_vec <- as.vector(as.matrix(X) %*% as.numeric(bet[names(X)]))

  # Give biomarker name of choice
  out <- data.frame(matrix(nrow = nrow(data), ncol = 0), check.names = FALSE)

  # Attach sample id column if provided; if NULL, use row names of input data
  if (!is.null(id_col)) {
    j <- ci_match(id_col, names(data))
    if (is.na(j)) stop("`id_col` not found (case-insensitive): ", id_col)
    ids <- data[[j]]
  } else {
    ids <- rownames(data)
    if (is.null(ids) || length(ids) == 0) {
      ids <- as.character(seq_len(nrow(data)))  # alternative when row names are absent, row numbers
    }
  }
  out[[id_name]] <- ids

  # Attach score column with the exact biomarker name
  out[[score_col]] <- score_vec

  rownames(out) <- NULL
  out
}

# Also create functions that provide ProtFI and ProtMort
#' @rdname protpredict
#' @export
predictProtFI <- function(data, id_col = NULL, scale = FALSE,
                          coef_df = NULL, highcor_df = NULL) {
  protpredict(
    data       = data,
    biomarker  = "ProtFI",
    id_col     = id_col,
    id_name    = "sampleid",
    scale      = scale,
    coef_df    = coef_df,
    highcor_df = highcor_df
  )
}

#' @rdname protpredict
#' @export
predictProtMort <- function(data, id_col = NULL, scale = FALSE,
                          coef_df = NULL, highcor_df = NULL) {
  protpredict(
    data       = data,
    biomarker  = "ProtMort",
    id_col     = id_col,
    id_name    = "sampleid",
    scale      = scale,
    coef_df    = coef_df,
    highcor_df = highcor_df
  )
}
