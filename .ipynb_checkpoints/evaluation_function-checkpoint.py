# load functions
import numpy as np
import pandas as pd
import random
import seaborn as sns 
import os
import forestplot as fp
import re
import scipy.stats as stats


random.seed(7)

from src.utils import get_data, get_nn_output
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functools import reduce
from lifelines import CoxPHFitter
from statsmodels.api import OLS, Logit, add_constant
from statsmodels.stats.weightstats import ztest
from scipy.stats import shapiro, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations

def check_significance(array1, array2):
    """
    Compares two paired datasets using either a Z-test (if normally distributed)
    or a Wilcoxon signed-rank test (if not). Also returns appropriate summary statistics.
    
    Parameters:
        array1 (numpy array): First dataset.
        array2 (numpy array): Second dataset.
    
    Returns:
        stats (float): Test statistic.
        p_value (float): P-value of the test.
        summary (dict): Summary statistics including mean/SD or median/IQR.
    """
    # Ensure arrays are NumPy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # Compute differences
    differences = array1 - array2
    
    # Check normality of the differences
    stat, p_value_normality = shapiro(differences)
    
    if p_value_normality > 0.05:
        print("Differences are normally distributed. Using Z-test.")
        stats, p_value = ztest(array1, array2, value=0, usevar='unequal') 
        
        # Compute mean and SD
        mean_diff = np.mean(differences)
        sd_diff = np.std(differences, ddof=1)
        summary = {"Mean Difference": mean_diff, "SD": sd_diff}
    
    else:
        print("Differences are not normally distributed. Using Wilcoxon signed-rank test.")
        stats, p_value = wilcoxon(array1, array2)
        
        # Compute median and IQR
        median_diff = np.median(differences)
        q1, q3 = np.percentile(differences, [25, 75])
        iqr = q3 - q1
        summary = {"Median Difference": median_diff, "IQR": iqr}
    
    if p_value < 0.05:
        print("The performance metrics are significantly different.")
    else:
        print("No significant difference in performance metrics.")
    
    return stats, p_value, summary

def densityplot(array1, array2, name1, name2, custom_palette, filename):
    """
    Creates a density plot to visually compare the distribution of two metric arrays 
    (e.g., R² values or concordance indices) corresponding to different biomarkers or models.

    The function labels and combines both arrays, infers an appropriate x-axis label,
    and generates a filled density plot using.
    The figure is saved in EPS format with rasterization enabled.

    Parameters:
        array1 (array-like): First set of metric values (e.g., performance scores).
        array2 (array-like): Second set of metric values.
        name1 (str): Label for the first set of metrics, used in the legend.
        name2 (str): Label for the second set of metrics.
        custom_palette (dict): Dictionary specifying colors for each label.
        filename (str): Path where the resulting EPS figure will be saved.

    Returns:
        None. Saves and displays the density plot.
    """
    df1 = pd.DataFrame({'Metric': array1, 'Biomarker': name1})
    df2 = pd.DataFrame({'Metric': array2, 'Biomarker': name2})
    difsdf = pd.concat([df2, df1])
    
    # Determine x-axis label based on the names
    if name1.startswith('R2') or name2.startswith('R2'):
        xlabel = r'$R^2$'
    elif name1.startswith('C') or name2.startswith('C'):
        xlabel = 'Concordance Index'
    else:
        xlabel = 'Metric'  # Default if neither condition is met

    # Create density plot
    fig, ax = plt.subplots()
    difplot = sns.kdeplot(data=difsdf, x="Metric", hue="Biomarker", fill=True, common_norm=False, alpha=0.6, palette=custom_palette, ax=ax)
    
    # Set rasterization for transparency handling
    ax.set_rasterized(True)

    # Move legend
    sns.move_legend(
        difplot, "lower center",
        bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
    )
    
    # Set x-axis label
    plt.xlabel(xlabel)
    
    # Save figure with rasterization
    plt.savefig(filename, format='eps', bbox_inches='tight')
    plt.show()

def calculate_biomarkers(protein_df, coefficients_df, name_col, weight_col):
    """
    Calculates biomarkers based on Z-sclaed NPX values and corresponding coefficients

    Parameters:
        protein_df (DataFrame): DataFrame where rows are participants and columns are protein names.
        coefficients_df (DataFrame): DataFrame containing protein names and their corresponding weights.
        name_col (str): Column name in coefficients_df with protein names.
        weight_col (str): Column name in coefficients_df with coefficients.

    Returns:
        Series: A pandas Series with the biomarker score for each participant.
    """
    # Map proteins to coefficients (protein names are keys, coefficients are values)
    coefficient_dict = coefficients_df.set_index(name_col)[weight_col].to_dict()
    
    # Identify the common proteins
    common_proteins = protein_df.columns.intersection(coefficient_dict.keys())
    
    # Multiply protein by weight
    weighted_values = protein_df[common_proteins].mul([coefficient_dict[protein] for protein in common_proteins], axis=1)
    
    # Create sumscore
    sumscore = weighted_values.sum(axis=1)
    
    return sumscore


def make_pandas_from_data(dset, target):
    """
    Returns a DataFrame with protein values and UKBB participant IDs 
    for the given dataset and training target, using `get_data()`.

    Parameters:
        dset (str): Protein set.
        target (str): Training outcome.

    Returns:
        DataFrame: Protein values with 'eid' as index and column.
    """

    _, set3prots, _, set3eids = get_data({'dset': dset, 'target': target, 'add_age':0, 'combine_sets':True})
    _, _, _, _, _, _, cols = get_data({'dset': dset, 'target': target, 'add_age':0, 'combine_sets':False})
    pandasframe = pd.DataFrame(set3prots[0], columns = cols[:-1], index = set3eids)
    pandasframe['eid'] = pandasframe.index
    return pandasframe

def process_en_models(directory, target, set3_df):
    """
    Create all Elastic Net biomarkers in a directory by computing weighted biomarker scores 
    and merging them into a shared DataFrame.

    Parameters:
        directory (str): Path to the folder containing model weight CSV files.
        target (str): Training outcome 'frailty' or 'mort'.
        set3_df (DataFrame): Base DataFrame with participant IDs ('eid') to merge results into.

    Returns:
        DataFrame: Updated DataFrame with biomarker scores from all models added as new columns.
    """
    # Loop through selected models directory
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_name = os.path.splitext(file)[0]  # remove .csv
            file_path = os.path.join(directory, file)
            weights_df = pd.read_csv(file_path)  # Load the weights DataFrame

            # Find the dset using regular expression
            match = re.search(rf'{target}_(.*?)_?(?:ffs)?$', file_name)
            if match:
                # Call function to create pandas DataFrame from the model type and matched group
                set3_model = make_pandas_from_data(match.group(1), target)
                set3_model[file_name] = calculate_biomarkers(set3_model, weights_df, weights_df.columns[0], weights_df.columns[1])

                # Merge with set3_df
                set3_df = pd.merge(set3_df, set3_model[['eid', file_name]], on='eid', how='outer')
            else:
                print(f'Not the right way to do this, error is for {target} and {file_name}!')
    return set3_df


def assign_nn_out(dataset_type, target, model_path):
    """
    Returns a neural network biomarker on the specified data set and assigns its output 
    as a DataFrame to a global variable named based on data set and traing target.

    Parameters:
        dataset_type (str): Name of the dataset (e.g., 'allprot', 'cmb').
        target (str): Training outcome, 'frailty' or 'mort'.
        model_path (str): Path to the trained neural network model.

    Returns:
        None. Stores the result as a global variable named 'nn_<dataset>_<target>'.
    """
    _, data_array,_,eids = get_data({'dset': dataset_type, 'target': target, 'combine_sets':True})
    var_name = f'nn_pred_{dataset_type}_{target}'
    df = get_nn_output(data_array[0], model_path, dataset_type, target, data_index = eids)
    df.columns = [f'nn_{dataset_type}_{target}']
    globals()[var_name] = df


def run_coxph_models(compare_df_keep, events, censorage, additional_columns, output_path, add_metabohealth=True, add_gdf15=True):
    """
    Runs Cox regression models for multiple biomarkers across different outcomes and covariate sets.
    Uses age-adjusted residuals (age acceleration) and optionally includes mortScore and GDF15.

    Parameters:
        compare_df_keep (DataFrame): Input dataframe with biomarkers and covariates.
        events (list of lists): Event variables per outcome.
        censorage (list of lists): Corresponding censoring ages per outcome.
        additional_columns (list of lists): Covariate sets to include.
        output_path (str): Path to save the output .csv.
        add_metabohealth (bool): Whether to include models additionally adjusting for MetaboHealth (mortScore).
        add_gdf15 (bool): Whether to include include models additionally adjusting for GDF15.

    Returns:
        DataFrame: Combined results with coefficients, HRs, p-values, C-index, and formatted strings.
    """
    
    cph_lifelines = CoxPHFitter()
    scaler = StandardScaler()

    if add_metabohealth:
        additional_columns += [['mortScore'] + x for x in additional_columns]
    if add_gdf15:
        additional_columns += [['GDF15'] + x for x in additional_columns]

    coefs = pd.DataFrame(columns=[
        'outcome', 'variable', 'model', 'covariates', 'N', 'N_event',
        'coef', 'se(coef)', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'C',
        'Biomarker2', 'Biomarker2_coef', 'Biomarker2_HR', 'Biomarker2_LL', 'Biomarker2_UL', 'Biomarker2_p'
    ])

    for column in compare_df_keep.columns:
        if column.startswith('coefs_') or column.startswith('nn_') or column == 'GDF15' or \
           column.startswith('Gadd') or column.startswith('ProteinAge') or column in ['PAC', 'GDF15', 'mortScore']:

            for o in range(len(events)):
                for i in range(len(additional_columns)):

                    if column == 'GDF15' and 'GDF15' in additional_columns[i]:
                        continue
                    if column == 'mortScore' and 'mortScore' in additional_columns[i]:
                        continue

                    subset_cols = [column] + additional_columns[i] + ['age'] + events[o] + censorage[o]
                    compare_df2 = compare_df_keep.dropna(subset=subset_cols)

                    model_data = compare_df2[subset_cols].copy()

                    if i != 0:
                        Xres = model_data['age'].values.reshape(-1, 1)
                        yres = model_data[column].values.reshape(-1, 1)
                        aa_model = LinearRegression().fit(Xres, yres)
                        model_data[[column]] = yres - aa_model.predict(Xres)

                    if 'mortScore' in additional_columns[i] or 'GDF15' in additional_columns[i]:
                        biomarker = 'mortScore' if 'mortScore' in additional_columns[i] else 'GDF15'
                        X_biomarker = model_data['age'].values.reshape(-1, 1)
                        y_biomarker = model_data[biomarker].values.reshape(-1, 1)
                        aa_model_biom = LinearRegression().fit(X_biomarker, y_biomarker)
                        residuals_biom = y_biomarker - aa_model_biom.predict(X_biomarker)
                        model_data[biomarker] = scaler.fit_transform(residuals_biom)

                    model_data[column] = scaler.fit_transform(model_data[[column]])
                    model_data.drop('age', axis=1, inplace=True)

                    cph_lifelines.fit(model_data, censorage[o][0], events[o][0])
                    summary = cph_lifelines.summary

                    coef_values = summary.loc[[column], [
                        'coef', 'se(coef)', 'exp(coef)', 'exp(coef) lower 95%',
                        'exp(coef) upper 95%', 'p']].copy()

                    coef_values['outcome'] = events[o][0]
                    coef_values['variable'] = f'aa_{column}'
                    coef_values['model'] = i
                    coef_values['N'] = len(cph_lifelines.durations)
                    coef_values['N_event'] = cph_lifelines.event_observed.sum()
                    coef_values['C'] = cph_lifelines.concordance_index_

                    coef_values['covariates'] = ', '.join(additional_columns[i]) if additional_columns[i] else 'None Not Age Accelerated'

                    if 'mortScore' in additional_columns[i]:
                        mh = summary.loc[['mortScore']]
                        coef_values['Biomarker2'] = 'mortScore'
                        coef_values['Biomarker2_coef'] = mh['coef'].iloc[0]
                        coef_values['Biomarker2_HR'] = mh['exp(coef)'].iloc[0]
                        coef_values['Biomarker2_LL'] = mh['exp(coef) lower 95%'].iloc[0]
                        coef_values['Biomarker2_UL'] = mh['exp(coef) upper 95%'].iloc[0]
                        coef_values['Biomarker2_p'] = mh['p'].iloc[0]
                    elif 'GDF15' in additional_columns[i]:
                        gdf = summary.loc[['GDF15']]
                        coef_values['Biomarker2'] = 'GDF15'
                        coef_values['Biomarker2_coef'] = gdf['coef'].iloc[0]
                        coef_values['Biomarker2_HR'] = gdf['exp(coef)'].iloc[0]
                        coef_values['Biomarker2_LL'] = gdf['exp(coef) lower 95%'].iloc[0]
                        coef_values['Biomarker2_UL'] = gdf['exp(coef) upper 95%'].iloc[0]
                        coef_values['Biomarker2_p'] = gdf['p'].iloc[0]
                    else:
                        for c in ['Biomarker2', 'Biomarker2_coef', 'Biomarker2_HR', 'Biomarker2_LL', 'Biomarker2_UL', 'Biomarker2_p']:
                            coef_values[c] = np.nan

                    coef_values = coef_values[[
                        'outcome', 'variable', 'model', 'covariates', 'N', 'N_event', 'coef', 'se(coef)',
                        'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'C',
                        'Biomarker2', 'Biomarker2_coef', 'Biomarker2_HR', 'Biomarker2_LL', 'Biomarker2_UL', 'Biomarker2_p'
                    ]]

                    coefs = pd.concat([coefs, coef_values], ignore_index=True)

    coefs['HR_CI'] = coefs.apply(lambda row: f"{round(row['exp(coef)'], 2)} ({round(row['exp(coef) lower 95%'], 2)}; {round(row['exp(coef) upper 95%'], 2)})", axis=1)
    coefs['Biomarker2_CI'] = coefs.apply(
        lambda row: f"{round(row['Biomarker2_HR'], 2)} ({round(row['Biomarker2_LL'], 2)}; {round(row['Biomarker2_UL'], 2)})"
        if pd.notna(row['Biomarker2']) else "", axis=1
    )
    coefs['Concordance'] = round(coefs['C'], 2)

    print("Final Coefficients DataFrame:")
    print(coefs)

    coefs.to_csv(output_path, index=False)
    return coefs

def run_linlog_models(compare_frailty, outcomes, frailty_additional_columns, output_path, add_metabohealth=True, add_gdf15=True, comparebiom=False):
    """
    Runs linear or logistic regression models for multiple frailty outcomes using biomarkers 
    and covariates, with age-adjusted residuals and optional inclusion of mortScore and GDF15.

    Parameters:
        compare_frailty (DataFrame): Input dataframe with biomarkers, outcomes, and covariates.
        outcomes (list of lists): List of outcome variable names (e.g. [['FI_0'], ['CVD_prev'], ...]).
        frailty_additional_columns (list of lists): Covariate sets to use.
        output_path (str): Path to save the resulting .csv.
        add_metabohealth (bool): Whether to include models additionally adjusting for MetaboHealth (mortScore).
        add_gdf15 (bool):  Whether to include models additionally adjusting for GDF15.

    Returns:
        DataFrame: Results dataframe with coefficients, p-values, and formatted strings.
    """
    scaler = StandardScaler()
    results = []
    comparebiomarkers = comparebiom

    # Extend additional columns if needed
    if add_metabohealth:
        frailty_additional_columns += [['mortScore'] + sub for sub in frailty_additional_columns]
    if add_gdf15:
        frailty_additional_columns += [['GDF15'] + sub for sub in frailty_additional_columns]

    for out in outcomes:
        for column in compare_frailty.columns:
            if column.startswith('coefs_') or column.startswith('nn_') or column == 'GDF15' or \
               column.startswith('Gadd') or column.startswith('ProteinAge') or column in ['PAC', 'GDF15', 'mortScore']:
                
                for i in range(len(frailty_additional_columns)):
                    if column == 'mortScore' and 'mortScore' in frailty_additional_columns[i]:
                        continue
                    if column == 'GDF15' and 'GDF15' in frailty_additional_columns[i]:
                        continue

                    # Drop based on biomarker availability
                    if comparebiomarkers:
                        biomarker_cols = [col for col in compare_frailty.columns if col.startswith(('coefs_', 'nn_', 'Gadd')) or col in ['PAC', 'GDF15', 'mortScore']]
                    else:
                        biomarker_cols = [col for col in compare_frailty.columns if col.startswith(('coefs_', 'nn_'))]

                    compare_frailty_keep = compare_frailty.dropna(subset=biomarker_cols)

                    if len(set(out).intersection(set(frailty_additional_columns[i]))) == 0:
                        frailty_add = list(np.append(out, frailty_additional_columns[i]))
                        model_data = compare_frailty_keep[[column] + frailty_add].copy()
                    else:
                        model_data = compare_frailty_keep[[column] + frailty_additional_columns[i]].copy()
                        model_data = model_data.reindex(columns=(out + [c for c in model_data.columns if c != out[0]]))

                    model_data.dropna(inplace=True)

                    if i != 0:
                        Xres = model_data['age'].values.reshape(-1, 1)
                        yres = model_data[column].values.reshape(-1, 1)
                        aa_model = LinearRegression().fit(Xres, yres)
                        model_data[[column]] = yres - aa_model.predict(Xres)

                        if 'mortScore' in frailty_additional_columns[i]:
                            X = model_data['age'].values.reshape(-1, 1)
                            y = model_data[['mortScore']]
                            res = y - LinearRegression().fit(X, y).predict(X)
                            model_data[['mortScore']] = scaler.fit_transform(res)

                        elif 'GDF15' in frailty_additional_columns[i]:
                            X = model_data['age'].values.reshape(-1, 1)
                            y = model_data[['GDF15']]
                            res = y - LinearRegression().fit(X, y).predict(X)
                            model_data[['GDF15']] = scaler.fit_transform(res)

                    model_data[[column]] = scaler.fit_transform(model_data[[column]])
                    model_data.drop(columns=['age'], inplace=True, errors='ignore')

                    if out[0] in [x[0] for x in outcomes[:len(outcomes) - 2]]:  # Assuming last 2 are risk factors
                        X = model_data.drop(columns=out[0])
                        y = model_data[out]
                    else:
                        X = model_data.drop(columns=column)
                        y = model_data[column]

                    try:
                        if len(np.unique(y.values)) > 2:
                            an_type = 'linear'
                            reg = OLS(y, add_constant(X)).fit()
                            r_squared = reg.rsquared
                            coef = reg.params.iloc[1]
                            std_err = reg.bse.iloc[1]
                            conf_int_025, conf_int_975 = reg.conf_int().iloc[1]
                            Ncase = None
                            t_val = reg.tvalues.iloc[1]
                            p_val = reg.pvalues.iloc[1]

                        else:
                            an_type = 'logistic'
                            reg = Logit(y, add_constant(X)).fit()
                            coef = np.exp(reg.params.iloc[1])
                            std_err = np.exp(reg.bse.iloc[1])
                            conf_int_025, conf_int_975 = np.exp(reg.conf_int().iloc[1])
                            Ncase = y[out].sum()
                            r_squared = reg.prsquared
                            t_val = reg.tvalues.iloc[1]
                            p_val = reg.pvalues.iloc[1]

                        if 'mortScore' in frailty_additional_columns[i] or 'GDF15' in frailty_additional_columns[i]:
                            idx = 2  # Biomarker2 is the third variable
                            biomarker2 = 'MetaboHealth' if 'mortScore' in frailty_additional_columns[i] else 'GDF15'
                            if an_type == 'linear':
                                Biomarker2_coef = reg.params.iloc[idx]
                                Biomarker2_LL, Biomarker2_UL = reg.conf_int().iloc[idx]
                                Biomarker2_p = reg.pvalues.iloc[idx]
                            else:
                                Biomarker2_coef = np.exp(reg.params.iloc[idx])
                                Biomarker2_LL, Biomarker2_UL = np.exp(reg.conf_int().iloc[idx])
                                Biomarker2_p = reg.pvalues.iloc[idx]
                        else:
                            biomarker2 = Biomarker2_coef = Biomarker2_LL = Biomarker2_UL = Biomarker2_p = np.nan

                        covar = 'None Not Age Accelerated' if i == 0 else ', '.join([c for c in frailty_additional_columns[i] if c != 'age'])

                        results.append({
                            'type': an_type,
                            'outcome': out[0],
                            'variable': f'aa_{column}',
                            'model': i,
                            'N': len(y),
                            'Ncase': Ncase,
                            'Coef': coef,
                            'Std Err': std_err,
                            't-value': t_val,
                            'P': p_val,
                            'LL': conf_int_025,
                            'UL': conf_int_975,
                            'R-squared': r_squared,
                            'Biomarker2': biomarker2,
                            'Biomarker2_coef': Biomarker2_coef,
                            'Biomarker2_LL': Biomarker2_LL,
                            'Biomarker2_UL': Biomarker2_UL,
                            'Biomarker2_p': Biomarker2_p,
                            'CompareExternal': comparebiomarkers,
                            'Covariates': covar
                        })

                    except Exception as e:
                        print(f"Model failed for {column} ~ {out[0]} with model {i}: {e}")
                        continue

    results_df = pd.DataFrame(results)
    results_df['Beta_CI'] = results_df.apply(lambda row: f"{round(row['Coef'], 2)} ({round(row['LL'], 2)}; {round(row['UL'], 2)})", axis=1)
    results_df['Biomarker2_CI'] = results_df.apply(
        lambda row: f"{round(row['Biomarker2_coef'], 2)} ({round(row['Biomarker2_LL'], 2)}; {round(row['Biomarker2_UL'], 2)})"
        if pd.notna(row['Biomarker2']) else "", axis=1
    )
    results_df['R2'] = round(results_df['R-squared'], 2)

    print(results_df)
    results_df.to_csv(output_path, index=False)
    return results_df
