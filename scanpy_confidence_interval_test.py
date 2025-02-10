import numpy as np
from utils import get_scanpy_lfcs, get_LN_lfcs, get_new_seurat_lfcs
import matplotlib.pyplot as plt

"""
Scanpy function to get mean and vars used for computing the t statistic

def _get_mean_var(
    X: _SupportedArray, *, axis: Literal[0, 1] = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(X, sparse.spmatrix):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = axis_mean(X, axis=axis, dtype=np.float64)
        mean_sq = axis_mean(elem_mul(X, X), axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var
"""

np.random.seed(0)
log_batch_factor = 0
nx = 1000
ny = 1000
n_genes = 1500
non_de_mu = 10

z1 = np.random.binomial(1, 0.1, (1, n_genes))
if non_de_mu == 10:
    mu1 = non_de_mu + np.abs(np.random.normal(0, 5, (1, n_genes))) * z1
else:
    mu1 = non_de_mu + np.random.normal(15, 5, (1, n_genes)) * z1
mu1 = np.tile(mu1, (nx, 1))
mu2 = non_de_mu * np.ones((ny, n_genes))
# the first half of all samples are batch effected
mu1[:nx // 2] *= np.exp(log_batch_factor)
mu2[:ny // 2] *= np.exp(log_batch_factor)

d1 = 2.
d2 = 1

r1 = 1 / d1
r2 = 1 / d2
p1 = 1 / (1 + d1 * mu1)
p2 = 1 / (1 + d2 * mu2)

# Generate synthetic gene expression data
X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

# remove batch effect for visualization purposes and ground truth
mu1[:nx // 2] *= np.exp(-log_batch_factor)
mu2[:ny // 2] *= np.exp(-log_batch_factor)
# mu sequences are now identical across samples within group
true_lfcs = np.log2(mu2[0] / mu1[0]).reshape((1, n_genes))
non_de_idx = (true_lfcs == 0.).reshape(-1)


sc_lfcs = get_scanpy_lfcs(X, Y, normalize=True)
se_lfcs = get_new_seurat_lfcs(X, Y, normalize=True)
X_raw, Y_raw = X.copy(), Y.copy()
# this is how Scanpy calculates the statistics used in the t test (based on their code)
# the results here were reproduced in the debugging console within the scanpy code, too
X = np.log(1e4 * X / (X.sum(1, keepdims=True)) + 1)
Y = np.log(1e4 * Y / (Y.sum(1, keepdims=True)) + 1)

mean_Y, mean_X = np.mean(Y, axis=0), np.mean(X, axis=0)

mean_sq_Y, mean_sq_X = np.mean(Y * Y, axis=0), np.mean(X * X, axis=0)
var_Y = (mean_sq_Y - mean_Y ** 2) * ny / (ny - 1)
var_X = (mean_sq_X - mean_X ** 2) * nx / (nx - 1)

se_Y = np.sqrt(var_Y / ny)
se_X = np.sqrt(var_X / nx)
se = np.sqrt(se_Y ** 2 + se_X ** 2) / np.log(2)
# for simplicity, the confidence intervals are formed with a critical z value (not t)
# but for these sample sizes the values are similar
confidence_intervals = (mean_Y - mean_X) / np.log(2) + 1.96 * np.array([-se, se])
for lfcs in [sc_lfcs, se_lfcs]:
    print(f"Estimated LFC is below CI {np.sum(lfcs[~non_de_idx] < confidence_intervals[0, ~non_de_idx])} times for DEGs")
    print(f"Estimated LFC is above CI {np.sum(lfcs[~non_de_idx] > confidence_intervals[1, ~non_de_idx])} times for DEGs")
    print()
    print(f"Estimated LFC is below CI {np.sum(lfcs[non_de_idx] < confidence_intervals[0, non_de_idx])} times for non-DEGs")
    print(f"Estimated LFC is above CI {np.sum(lfcs[non_de_idx] > confidence_intervals[1, non_de_idx])} times for non-DEGs")
    print(f"Frequency of estimated LFC that are above CI {np.sum(lfcs[non_de_idx] > confidence_intervals[1, non_de_idx]) / np.sum(non_de_idx)} for  non-DEGs")
    print()
plt.rcParams.update({'font.size': 20})
plt.errorbar(true_lfcs[0], (mean_Y - mean_X) / np.log(2), yerr=1.96 * se, fmt='none', label='Estimated CIs', color='green')
plt.plot(true_lfcs[0], sc_lfcs, 'o', label=r'$S_1$')
plt.plot(true_lfcs[0], se_lfcs, 'o', label=r'$S_3$', color='black')
plt.plot(true_lfcs[0], true_lfcs[0], 'o', label='True LFCs')
plt.xlabel('LFCs')
plt.ylabel('LFCs')
plt.title('Scanpy and Seurat $t$-test')
plt.ylim(-2., 0.8)
plt.legend(loc='lower right', fontsize=15)
plt.yticks([-2.0, -1.0, -1.5, -0.5, 0.0, 0.5])
plt.xticks([-1.0, -0.5, 0.0])
plt.grid(True)
plt.tight_layout()
plt.show()

ln_lfcs, _, gamma = get_LN_lfcs(Y_raw, X_raw, return_standard_error=True)
# the gammas are aptly scaled to base-2 in the utils function
ln_confidence_intervals = ln_lfcs + 1.96 * np.array([-gamma, gamma])
print(f"LN's Estimated LFC is below CI {np.sum(ln_lfcs < ln_confidence_intervals[0])} times")
print(f"LN's Estimated LFC is above CI {np.sum(ln_lfcs > ln_confidence_intervals[1])} times")

plt.rcParams.update({'font.size': 20})
plt.errorbar(true_lfcs[0], ln_lfcs, yerr=1.96 * gamma, fmt='none', label='Estimated CIs', color='green')
plt.plot(true_lfcs[0], ln_lfcs, 'o', label=r'$S_\text{LN}$')
plt.plot(true_lfcs[0], true_lfcs[0], 'o', label='True LFCs')
plt.xlabel('LFCs')
plt.ylabel('LFCs')
plt.title("LN's $t$-test")
plt.ylim(-2., 0.8)
plt.legend(loc='lower right', fontsize=15)
plt.yticks([-2.0, -1.0, -1.5, -0.5, 0.0, 0.5])
plt.xticks([-1.0, -0.5, 0.0])
plt.grid(True)
plt.tight_layout()
plt.show()