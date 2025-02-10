# LNs-t-test
Code to reproduce results in the LN's t-test paper (currently under review). This is "research code", i.e. it's not a slick, well-oiled package (yet!), but the code indeed reproduces the results in our submission.

The code to run LN's t-test is found in utils.py and the function called get_LN_lfcs. The function takes two raw expression (n_cells x n_genes)-matrices, Y and X, and performs a DGE test comparing the two conditions. CP10K normalization was used to get all results in the submission but median-of-ratios is also implemented (following implementation instructions in the DESEQ-2 paper). If return_standard_error is set to True, \hat{gamma} is returned, too.

In the python files where experiments are setup, d1 refers to $\phi_X$ and d2 to $\phi_Y$. To reproduce the results in the tables in the submission, run large_scale_NB_latex_tables.py and nb_batch_test.R with the corresponding parameter combinations specified in the submission. Make sure that the results are stored as csv files in a common folder, and then run large_scale_NB_latex_tables.py to get the latex code for the experiment-specific table + box plots. 

The repo includes a requirements file that is not optimized/filtered. The results based on the python code was obtained using:
```
python 3.10.15
numpy 2.0.2
matplotlib 3.9.2
pandas 2.2.3
scanpy 1.10.3
scipy 1.14.1
sklearn 1.5.2
statsmodel 0.14.4
``` 
