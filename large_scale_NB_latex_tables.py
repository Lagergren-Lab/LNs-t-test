import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("scanpy_csv.csv")
df2 = pd.read_csv("seurat_csv.csv")

df1.drop(columns=['recall'], inplace=True)

df2.rename(columns={'acc': 'accuracy', 'prec': 'precision'}, inplace=True)
df2['method'] = df2['method'].apply(lambda x: 'Seurat ' + x)

df = pd.concat([df1, df2], ignore_index=True)
df['method'] = df['method'].apply(lambda x: x.replace('_', ' '))
df['method'] = df['method'].apply(lambda x: x.replace('Seurat t', 'Seurat t-test'))
df['method'] = df['method'].apply(lambda x: x.replace('LN', r"\underbar{LN's $t$-test}"))
df['method'] = df['method'].apply(lambda x: x.replace('Seurat wilcox', 'Seurat wilcoxon'))
df['method'] = df['method'].apply(lambda x: x.replace('wilcoxon', 'Wilcoxon'))
df['method'] = df['method'].apply(lambda x: x.replace('t-test', '$t$-test'))
df['dispersion'] = df['dispersion'].apply(lambda x: str(round(x, 1)))

# For the sample sizes used in the paper the overestim_var gav the same results as the standard t-test
df = df[df.method != 'Scanpy $t$-test overestim var']

df_avg = df.groupby(['method', 'dispersion']).mean().reset_index().drop(columns=['rep_no'])
print(df_avg.sort_values(by=["dispersion", 'method']).to_latex(index=False, float_format='%.3f'))

df['method'] = df['method'].apply(lambda x: x.replace(r"\underbar{LN's $t$-test}", "LN's $t$-test"))
df['method'] = df['method'].apply(lambda x: x.replace("Seurat Wilcoxon limma", "Wilcoxon limma"))

metrics = ['accuracy', 'precision', 'tpr', 'tnr', 'fpr', 'fnr', 'f1']
mnames = ['Accuracy', 'Precision', 'TPR', 'TNR', 'FPR', 'FNR', 'F1']
boxprops = dict(linewidth=3)
whiskerprops = dict(linewidth=3)
for dispersion in set(df.dispersion):
    ax = df[df.dispersion == dispersion].boxplot(column=metrics,
                                                 by='method', rot=90, fontsize=35,
                                                 layout=(1, len(metrics)), figsize=(40, 30), boxprops=boxprops,
                                                 whiskerprops=whiskerprops)
    plt.suptitle('$(\phi_{Y_j}, \phi_{X_j}) =$' + f' (0.1, {dispersion})', fontsize=35)
    for i, a in enumerate(ax):
        a.set_title(mnames[i], fontsize=40)
        a.set_xlabel('')
        a.set_xlabel('')
plt.show()

