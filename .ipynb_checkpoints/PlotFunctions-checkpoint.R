mutate_df_function = function(df){
    df = df %>%
    mutate(dataset = case_when(grepl('cmb_met_ffs', variable) ~ 'FFS on CMB + metabolites',
                           grepl('cmb_met', variable) ~ 'Cmb + metabolites',
                           grepl('cmb_ffs', variable) ~ 'FFS on CMB',
                           grepl('cmb', variable) ~ 'CMB',
                           grepl('allprot_ffs', variable) ~ 'FFS on allprot',
                           grepl('allprot', variable) ~ 'allprot'),
      trainedout = case_when(grepl('mort|Gadd|PAC|Grim', variable) ~ "All-cause mortality",
                             grepl('frailty', variable) ~ "Frailty index",
                            grepl('GDF', variable) ~ "GDF15",
                            grepl('annum|orvath|ProteinAge', variable) ~ 'Age'),
      Coef = case_when(outcome %in% c('FI_0', 'FI') ~ Coef * 100, 
                       outcome %in% c('bmi.0.0', 'bmi') ~ Coef * 5,
                       outcome == 'max_handgrip' ~ Coef * -1,
                       TRUE ~ Coef),
       LL = case_when(outcome %in% c('FI_0', 'FI') ~ LL * 100, 
                      outcome %in% c('bmi.0.0', 'bmi') ~ LL * 5,
                      outcome == 'max_handgrip' ~ LL * -1,
                      TRUE ~ LL),
       UL = case_when(outcome %in% c('FI_0', 'FI') ~ UL * 100, 
                      outcome %in% c('bmi.0.0', 'bmi') ~ UL * 5,
                      outcome == 'max_handgrip' ~ UL * -1,
                      TRUE ~ UL),
      Beta_CI = case_when(outcome %in% c('FI_0', 'FI', 'bmi.0.0', 'bmi', 'max_handgrip') ~ paste0(sprintf(Coef, fmt = "%.2f"), " (", sprintf(LL, fmt = "%.2f"), ";", sprintf(UL, fmt = "%.2f"), ")"),
                       TRUE ~ Beta_CI),
      outcome_name_use = case_when(grepl('FI', outcome) ~ "Frailty index (%)",
                               outcome == "max_handgrip" ~ "Grip strength decrease (kg)",
                               grepl('bmi', outcome) ~ 'BMI (per 5 kg/m²)',
                               grepl('smokingyn', outcome) ~ 'Current smoking',
                               grepl('hbp', outcome) ~ "High blood pressure",
                               outcome == 'cancer_prev' ~ 'Prevalent cancer',
                               outcome == 'CVD_prev' ~ 'Prevalent CVD',
                               outcome == 'poor_health' ~ 'Self-rated poor health',
                               outcome == "died" ~ "All-cause mortality",
                               outcome == "cancer_inc" ~ "Incident cancer",
                               outcome == "CVD_inc" | outcome == 'inc_cvd' ~ "Incident CVD"),
       trainmethod = case_when(grepl('coefs', variable) ~ "Elastic Net",
                               grepl('nn', variable) ~ "Feedforward\nneural network"),
       variable2 = paste0(dataset, "_", trainedout), 
       compare_var = recode(variable, 
                     aa_coefs_frailty_cmb = 'EN Frailty CMB',
                     aa_coefs_frailty_cmb_ffs = 'ProtFI',
                     aa_nn_cmb_frailty = 'FNN Frailty CMB',
                     aa_nn_cmb_ffs_frailty = 'FNN Frailty CMB_FFS',
                     aa_coefs_mort_cmb = 'EN Mortality CMB',
                     aa_coefs_mort_cmb_ffs = 'ProtMort',
                     aa_nn_cmb_mort = 'FNN Mortality CMB',
                     aa_nn_cmb_ffs_mort = 'FNN Mortality CMB_FFS',
                     aa_Gaddprot = "Protein Mortality Score",
                     aa_Gaddmet = "Metabo Mortality Score",
                     aa_PAC = "PAC 8 protein panels",
                     aa_GDF15 = "GDF15",
                     aa_mortScore = "MetaboHealth",
                     aa_Horvath = 'DNAm Horvath',
                     aa_DNAmHannumAge = 'DNAm Hannum',
                     aa_DNAmGrimAge = 'DNAm GrimAge',
                     aa_ProteinAge = 'ProteinAge',
                     aa_ProteinAge20 = 'ProteinAge20'),
          step2_var = case_when(compare_var %in% c('ProtFI', 'ProtMort') ~ 'ProtFI/ProtMort',
                                trainmethod == 'Elastic Net' & dataset == 'CMB' ~ 'EN CMB',
                                trainmethod == 'Feedforward\nneural network' & dataset == 'CMB' ~ 'FNN CMB',
                                trainmethod == 'Feedforward\neural network' & dataset == 'FFS on CMB' ~ 'FNN FFS on CMB')
          )
       return(df)}


plot_function <- function(df, x_name, vline_val, xlabs, levels_outcome, levels_var, levels_col, 
                          bgcol_val = c(1,2,3), biomarker_col_val = c(1:9),
                          show_trainedout_labels = FALSE, show_legend = TRUE, mode = "own_models") {
    
    # Filter and order the dataframe based on the selected mode
    if (mode == "own_models") {
        df = subset(df, outcome_name_use %in% levels_outcome & variable2 %in% levels_var)
        df$variable2 = factor(df$variable2, levels = rev(levels_var))
        y_variable = "variable2"
        df$trainmethod = factor(df$trainmethod, levels = rev(c('Elastic Net', 'Feedforward\nneural network')))
        df <- df %>% arrange(trainedout, trainmethod, dataset)
        df$color_var = df$dataset

        df <- df %>%
            mutate(group_id = interaction(trainedout, trainmethod)) %>%
            arrange(trainedout, trainmethod, dataset, variable2) %>%
            group_by(group_id) %>%
            mutate(row_in_group = row_number()) %>%
            ungroup()

        group_order <- df %>%
            distinct(group_id, trainedout, trainmethod) %>%
            arrange(trainedout, trainmethod) %>%
            mutate(group_index = row_number())

        df$group_index <- group_order$group_index[match(df$group_id, group_order$group_id)]

        df <- df %>%
            arrange(group_index, row_in_group) %>%
            mutate(numrow = row_number() + 2 * (group_index - 1))
                
    } else if (grepl("compare", mode)) {
        df = subset(df, outcome_name_use %in% levels_outcome & compare_var %in% levels_var)
        df$compare_var = factor(df$compare_var, levels = rev(levels_var))
        df$color_var = df$compare_var
        df$trainedout = ifelse(df$trainedout == 'Frailty index', 'FI', df$trainedout)
        df$trainedout = factor(df$trainedout, levels = rev(c('FI', 'All-cause mortality','Age', 'GDF15')))
        y_variable = "compare_var"
        if(mode == 'compare_voila'){
            df$Cohort = factor(df$Cohort, levels = rev(c('RS', 'LLS')))
            df <- df %>% arrange(trainedout, compare_var, Cohort)
        } else {
            df <- df %>% arrange(trainedout, compare_var)
        } 
        df$numrow <- 1:nrow(df)
    } else {
        stop("Invalid mode. Use 'own_models', 'compare' or 'compare_voila'.")
    }

    df['coef'] = df[[x_name]]
    df$color_var = factor(df$color_var, levels = rev(levels_col)) 

    if (mode == "own_models") {
    first_group <- df$trainedout[df$numrow == 1]
    other_group <- setdiff(unique(df$trainedout), first_group)

    halfway_point <- floor(max(df$numrow) / 2)
    trainedout_positions <- data.frame(
        trainedout = c(first_group, other_group),
        y_min = c(1, halfway_point + 1),
        y_max = c(halfway_point, max(df$numrow)),
        fill_var = c(first_group, other_group)
    )}
    else{
        trainedout_positions <- df %>%
        group_by(trainedout) %>%
        summarize(y_min = min(numrow), y_max = max(numrow), .groups = "drop") %>%
        mutate(fill_var = trainedout)
    }

    labels_map <- c('FI' = 'Frailty index', 
                    'Frailty index' = 'Frailty index',
                    'All-cause mortality' = 'All-cause mortality', 
                    'Age' = 'Chronological age', 
                    'GDF15' = 'Not trained')
    filtered_labels <- labels_map[as.character(unique(df$trainedout))]

    # Start plot
    plot <- ggplot(df, aes(x = coef, y = numrow, xmin = LL, xmax = UL, color = color_var)) +
    geom_rect(data = trainedout_positions,
                    aes(xmin = -Inf, xmax = Inf, ymin = y_min - 0.5, ymax = y_max + 0.5, fill = fill_var),
                      inherit.aes = FALSE, alpha = 0.3) +
    geom_point(size = 5, shape = 16)  
    # Add error bars
    if (mode == 'own_models') {
        plot <- plot + geom_errorbarh(aes(linetype = trainmethod), height = 0.1, linewidth = 2.5) +
               scale_linetype_manual(name = "Input dataset",
                                     values = c("Elastic Net" = "solid", 
                                                "Feedforward\nneural network" = "dotted"))
    } else if (mode == "compare_voila") {
        plot <- plot +
        geom_errorbarh(aes(linetype = Cohort), height = 0.1, linewidth = 2.5) +
        scale_linetype_manual(name = "Validation cohort",
                          values = c("RS" = "solid", 
                                     "LLS" = "dotted"))

    } else {
        plot <- plot + geom_errorbarh(height = 0.1, linewidth = 2.5, linetype = 'solid')
    }

    # Add titles, theme, etc.
    plot <- plot +
      labs(x = xlabs, y = NULL, title = paste("  ", levels_outcome)) +
      geom_vline(xintercept = vline_val, color = 'black', linetype = 'dashed', alpha = 0.5) +
      theme_minimal() +
      scale_fill_manual(name = "Training\noutcome", 
                        values = Tol_muted[bgcol_val],
                        breaks = unique(df$trainedout),
                        labels = filtered_labels) +
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            text = element_text(size = 18),
            plot.title = element_text(size = 22),
            legend.text = element_text(size = 20))

    # Color scheme
    if (mode %in% c("own_models", 'step2')) {
        plot <- plot + scale_color_manual(values = Tol_muted[biomarker_col_val], name = "Input\ndata", na.translate = FALSE)
    } else if (grepl("compare", mode)) {
        plot <- plot + scale_color_manual(values = colorblind[biomarker_col_val], name = "Biomarker", na.translate = FALSE)
    }

    # Legend controls
    if (show_legend) {
        plot <- plot + 
          guides(colour = guide_legend(order = 1, nrow = length(biomarker_col_val), byrow = TRUE, reverse = TRUE),
                 fill = guide_legend(order = 2, nrow = length(bgcol_val), byrow = TRUE, reverse = TRUE))
        if (mode %in% c("own_models" , "compare_voila")){
             plot <- plot + 
               guides(linetype = guide_legend(order = 3, nrow = 2, byrow = TRUE, reverse = TRUE))
         }
    } else {
         plot <- plot + guides(colour = 'none', fill = 'none', linetype = 'none')
    }

    # Labels for background
    if (show_trainedout_labels) {
        plot <- plot +
          geom_text(data = trainedout_positions,
                    aes(x = vline_val - (max(df$coef)/8), 
                        y = (y_min + y_max) / 2, label = trainedout),
                    inherit.aes = FALSE, hjust = 0.5, vjust = 0.5, angle = 90, size = 6) +  
          geom_text(data = trainedout_positions,
                     aes(x = vline_val - (max(df$coef)/4), 
                         y = mean(y_min + y_max)/2), 
                     label = "Trained on", inherit.aes = FALSE, 
                     hjust = 0.5, vjust = 0.5, angle = 90, size = 7, fontface = "bold") 
    }

    return(plot)
}

# Function to create plots
create_plot <- function(df, x_name, vline_val, xlabs, levels_outcome, levels_var, biomarker_col_val, bgcol_val, show_trainedout_labels, show_legend = TRUE, mode = "own_models") {
  if(identical(levels_var, var_allprot_cmb_ffs)){
      levels_colors = lev_col_4prot_cmb_ffs
  }else if(grepl('allprot', levels_var[1])){
      levels_colors = lev_col_4prot_cmb
  }else if(grepl('CMB', levels_var[1])){
      levels_colors = lev_col_cmb_ffs
  }else if(mode == 'step2'){
      levels_colors = lev_step2
  }else{
      levels_colors = levels_var
  }
    plot_function(
    df = df, x_name = x_name, vline_val = vline_val, xlabs = xlabs, 
    levels_outcome = levels_outcome, levels_var = levels_var, levels_col = levels_colors,
    biomarker_col_val = biomarker_col_val, bgcol_val = bgcol_val, 
    show_trainedout_labels = show_trainedout_labels, show_legend = show_legend, mode = mode
  )
}

should_show_legend <- function(outcome, name) {
  base_legend <- (outcome %in% c("Frailty index (%)", "All-cause mortality"))
  if (grepl("VOILA", name)) !base_legend else base_legend
}

validate_plot_configs <- function(plot_configs) {
  required_fields <- c("name", "df", "x", "v", "xl", "out", "var", "col", "showlab")
  valid <- logical(length(plot_configs))
  
  for (i in seq_along(plot_configs)) {
    cfg <- plot_configs[[i]]
    missing <- setdiff(required_fields, names(cfg))
    if (length(missing) > 0) {
      message("Config ", i, " ('", cfg$name, "') is missing fields: ", paste(missing, collapse = ", "))
      valid[i] <- FALSE
    } else {
      valid[i] <- TRUE
    }
  }
  
  return(valid)
}

