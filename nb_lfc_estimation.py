import numpy as np
from utils import get_LN_lfcs, get_scanpy_lfcs, get_new_seurat_lfcs
import matplotlib.pyplot as plt

np.random.seed(0)

nx = 1000
ny = 1000
n_genes = 1500

# dispersions
non_de_mu = 10
if non_de_mu == 10:
    d2 = 1.
    d1_list = np.arange(10, 21) / 10
else:
    d2 = 0.1
    d1_list = np.arange(1, 11) / 10
results = {r"$S_\text{LN}$": np.zeros((2, len(d1_list))),
           "$S_1$": np.zeros((2, len(d1_list))),
           "$S_3$": np.zeros((2, len(d1_list))),
           "$t$-test": np.zeros((2, len(d1_list)))}
for lam_idx, log_batch_factor in enumerate([0., 1.]):
    for d_idx, d1 in enumerate(d1_list):
        # with probability 0.1, there is Gaussian noise added to a gene in group 1
        rep_count = 20
        for rep in range(rep_count):
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

            r1 = 1 / d1
            r2 = 1 / d2
            p1 = 1 / (1 + d1 * mu1)
            p2 = 1 / (1 + d2 * mu2)

            X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
            Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

            # remove batch effect for visualization purposes and ground truth
            mu1[:nx // 2] *= np.exp(-log_batch_factor)
            mu2[:ny // 2] *= np.exp(-log_batch_factor)
            true_lfcs = np.log2(mu2[0] / mu1[0]).reshape((1, n_genes))

            for method in [r"$S_\text{LN}$", "$t$-test", "$S_1$", "$S_3$"]:
                if method == r"$S_\text{LN}$":
                    lfcs, _ = get_LN_lfcs(Y, X, test='t')

                elif method == "$S_1$":
                    lfcs = get_scanpy_lfcs(X, Y)

                elif method == "$S_3$":
                    lfcs = get_new_seurat_lfcs(X, Y)

                else:
                    log_X = np.log(1e4 * X / (X.sum(1, keepdims=True)) + 1)
                    log_Y = np.log(1e4 * Y / (Y.sum(1, keepdims=True)) + 1)

                    mu_tilde_Y, mu_tilde_X = np.mean(log_Y, axis=0), np.mean(log_X, axis=0)
                    lfcs = (mu_tilde_Y - mu_tilde_X) / np.log(2)

                results[method][lam_idx, d_idx] += np.sqrt(np.mean((lfcs - true_lfcs) ** 2)) / rep_count

plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(15, 10))
for method, color in zip([r"$S_\text{LN}$", "$t$-test", "$S_1$", "$S_3$"], ["blue", "green", "red", "black"]):
    plt.plot(d1_list, results[method][0], label=method, ls="dashed", color=color)
    plt.plot(d1_list, results[method][1], color=color)
    print(method)
    print(results[method][0])
    print(results[method][1])
plt.legend(fontsize=20)
if non_de_mu == 100:
    plt.title("Densely Expressed Gene Data")
elif non_de_mu == 10:
    plt.title("Sparsely Expressed Gene Data")
plt.xlabel(r"$\phi_{X_j}$")
plt.ylabel(r"RMSE")
plt.xticks(d1_list)
plt.yticks(np.arange(1, 7) / 10)
plt.show()