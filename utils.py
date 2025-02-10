import numpy as np
import scipy.stats as stats
from scipy.stats.distributions import t
from sklearn.metrics import confusion_matrix
import scanpy as sc
import pandas as pd


def scanpy_sig_test(X, Y, method='t-test', normalization='CP10K'):
    nx = X.shape[0]
    ny = Y.shape[0]
    n_genes = Y.shape[1]
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    if normalization == 'median-of-ratios':
        X = X.astype(float)
        Y = Y.astype(float)
        X_ = X.copy()
        Y_ = Y.copy()
        X_[X_ <= 0] = np.nan
        Y_[Y_ == 0] = np.nan

        denom_Y = np.exp(np.nanmean(np.log(Y_), 0))
        c_Y = np.nanmedian(Y_ / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X_), 0))
        c_X = np.nanmedian(X_ / denom_X, 1, keepdims=True)
        X /= c_X

        Y[Y == np.nan] = 0.
        X[X == np.nan] = 0.

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    adata.layers["counts"] = adata.X.copy()
    if normalization == 'CP10K':
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy's t-test
    corr_method = "bonferroni"  # use Bonferroni to match output of Seurat
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X", corr_method=corr_method)

    # Extract DE Results
    de_results = pd.DataFrame({
        "gene": adata.uns['rank_genes_groups']['names']['Y'],
        "log2_fc": adata.uns["rank_genes_groups"]["logfoldchanges"]["Y"],
        "p_value": adata.uns["rank_genes_groups"]["pvals"]["Y"],
        "p_adj": adata.uns["rank_genes_groups"]["pvals_adj"]["Y"]
    })
    de_results["gene"] = [int(gene.replace("Gene", "")) for gene in de_results["gene"]]
    gene_idx_sorted = np.argsort(de_results["gene"])

    return de_results["log2_fc"][gene_idx_sorted], de_results["p_adj"][gene_idx_sorted]


def get_test_results(adj_p_vals, true_lfcs, verbose=True):
    gt_sig_idx = (true_lfcs != 0).reshape((-1,))
    pred_sig_idx = (adj_p_vals < 0.05)
    conf_mat = confusion_matrix(gt_sig_idx, pred_sig_idx)
    tn, fp, fn, tp = conf_mat.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.sum((gt_sig_idx == pred_sig_idx)) / gt_sig_idx.size

    if verbose:
        # Print results
        print(f"TPR: {tpr:.2f}")
        print(f"TNR: {tnr:.2f}")
        print(f"FPR: {fpr:.2f}")
        print(f"FNR: {fnr:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print()
    results = {
        "f1": f1, "accuracy": accuracy, "recall": recall, "precision": precision,
        "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr
    }
    return results


def trigamma(x):
    return 1 / x  # + 0.5 / (x ** 2)

def get_LN_lfcs(Y_, X_, normalize=True, test='t', normalization='CP10K', return_standard_error=False):
    # Y is (n_cells, n_genes)
    eps = 1e-9

    Y = Y_.astype(float).copy()
    n = Y.shape[0]
    Y[Y <= 0] = np.nan  # Replace all non-positive with NaN
    n_plus = n - np.sum(np.isnan(Y), 0)

    X = X_.astype(float).copy()
    n_prime = X.shape[0]
    X[X <= 0] = np.nan
    n_plus_prime = n_prime - np.sum(np.isnan(X), 0)

    if normalize and (normalization == 'CP10K'):
        X = 1e4 * X / np.nansum(X, 1, keepdims=True)
        Y = 1e4 * Y / np.nansum(Y, 1, keepdims=True)

    elif normalize and (normalization == 'median-of-ratios'):
        # the normalization scheme proposed in DESeq2
        denom_Y = np.exp(np.nanmean(np.log(Y), 0))
        denom_Y[np.isnan(denom_Y)] = 1  # Avoid division by NaN for unexpressed genes
        c_Y = np.nanmedian(Y / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X), 0))
        denom_X[np.isnan(denom_X)] = 1  # Avoid division by NaN for unexpressed genes
        c_X = np.nanmedian(X / denom_X, 1, keepdims=True)
        X /= c_X


    pos_mean_Y = np.nanmean(Y, axis=0)
    pos_mean_X = np.nanmean(X, axis=0)

    # \hat{a}
    a_hat_Y = n_plus + (eps ** (1 + n_plus))
    a_hat_X = n_plus_prime + (eps ** (1 + n_plus_prime))

    # compute \log2\hat{theta} for each gene
    log2_theta_hat_Y = np.log2(a_hat_Y / n)
    log2_theta_hat_X = np.log2(a_hat_X / n_prime)

    # compute sample mean of positive counts
    log2_m_Y = np.log2(pos_mean_Y)
    log2_m_X = np.log2(pos_mean_X)

    lfc = (log2_theta_hat_Y + log2_m_Y) - (log2_theta_hat_X + log2_m_X)

    # compute squared (!) standard errors
    se_Y_1 = trigamma(a_hat_Y) - trigamma(n)  # squared SE for log theta_Y estimator
    se_Y_2 = np.log(1 + np.nanvar(Y, axis=0) / (n_plus * (2 ** log2_m_Y) ** 2))  # squared SE for log m_Y
    se_Y = np.sqrt(se_Y_1 + se_Y_2) / np.log(2)

    # compute squared (!) standard errors
    se_X_1 = trigamma(a_hat_X) - trigamma(n_prime)  # squared SE for log theta_X estimator
    se_X_2 = np.log(1 + np.nanvar(X, axis=0) / (n_plus_prime * (2 ** log2_m_X) ** 2))  # squared SE for log m_X
    se_X = np.sqrt(se_X_1 + se_X_2) / np.log(2)

    gamma = np.sqrt(se_X ** 2 + se_Y ** 2)

    if test == 't':
        statistic, p_vals = get_t_statistic(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X,
                                            se_Y, se_X, n, n_prime)
    else:
        # z-test
        statistic, p_vals = compute_p_vals(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X, se_Y, se_X)

    if return_standard_error:
        return lfc, p_vals, gamma
    return lfc, p_vals


def get_seurat_lfcs(X, Y, normalize=True):
    # NOT USED (see function below)
    # Manual calculation of the LFC based on how seurat implements it.
    # See Log fold-change calculation methods in https://www.biorxiv.org/content/10.1101/2022.05.09.490241v2.full.pdf
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2(np.mean(np.exp(log_Y) - 1, 0) + 1) - np.log2(np.mean(np.exp(log_X) - 1, 0) + 1)

def get_new_seurat_lfcs(X, Y, normalize=True, eps=1e-9):
    # Manual calculation of the LFC based on how seurat implements it.
    # See Log fold-change calculation methods in https://www.biorxiv.org/content/10.1101/2022.05.09.490241v2.full.pdf
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2((np.sum(np.exp(log_Y) - 1, 0) + eps) / Y.shape[0]) - np.log2((np.sum(np.exp(log_X) - 1, 0) + eps) / X.shape[0])


def get_scanpy_lfcs(X, Y, normalize=True):
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2(np.exp(np.mean(log_Y, 0)) - 1 + 1e-9) - np.log2(np.exp(np.mean(log_X, 0)) - 1 + 1e-9)


def transform(z):
    # Implements log1p(CP10k data)
    # log(10000 * z / z.sum(over genes for each cell) + 1)
    return np.log((z * 1e4 / z.sum(1, keepdims=True)) + 1)


def compute_p_vals(mean1, mean2, se1, se2):
    # z-test (t-test below)
    # Compute the test statistic
    z_stat = (mean1 - mean2) / ((se1 ** 2 + se2 ** 2) ** 0.5)

    # Compute the p-value for the two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value

def get_t_statistic(mean1, mean2, se1, se2, n1, n2):
    # implements two-sided t-test
    nu1, nu2 = n1 - 1, n2 - 1
    df = (se1 ** 2 + se2 ** 2) ** 2 / (se1 ** 4 / nu1 + se2 ** 4 / nu2)
    d = mean1 - mean2
    denom = ((se1 ** 2 + se2 ** 2) ** 0.5)
    t_statistic = d / denom
    t_dist = t(df)
    p_value = 2 * t_dist.sf(np.abs(t_statistic))
    return t_statistic, p_value