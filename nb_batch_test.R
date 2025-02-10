rm(list=ls())

# For running it on cluster -- call this script with path to the simulated data.
#args <- commandArgs(trailingOnly = T)
#base_path <- args[1]

library(data.table)
library(ggplot2)
library(Seurat)
source("function.R")

rep_count <- 20
mu = 100
base_path <- paste0("../simul/nb/batch_mu", mu)
dispersions <- c("0_1", "0_2", "1_0")
methods <- c("wilcox", "wilcox_limma", "t")

for (dispersion in dispersions)
{
  output_path <- paste0(base_path, "/dispersion_", dispersion)
  for (rep_no in seq(0, rep_count-1))
  {
    rep_path <- paste0(output_path, "/rep", rep_no)
    X <- read.table(paste0(rep_path, "/X.csv"), header = F)
    Y <- read.table(paste0(rep_path, "/Y.csv"), header = F)
    z <- as.numeric(read.table(paste0(rep_path, "/z.csv"), header = F))
    
    cts <- rbind(X, Y)
    cts <- t(cts)
    cell_count_x <- dim(X)[1]
    cell_count_y <- dim(Y)[1]
    n_genes <- dim(cts)[1]
    condition <- c(rep("X", cell_count_x), rep("Y", cell_count_y))
    seurat <- Seurat::CreateSeuratObject(counts = cts, assay = "RNA")
    seurat$cond <- condition
    
    seurat <- NormalizeData(seurat)
    seurat <- FindVariableFeatures(seurat)
    seurat <- ScaleData(seurat)
    #seurat <- RunPCA(seurat)
    #seurat <- RunUMAP(seurat, dims = 1:30)
    #DimPlot(seurat, reduction = "umap", group.by = "cond")
    
    id1 <- "X"
    id2 <- "Y"
    Idents(seurat) <- "cond"
    feature_names <- rownames(cts)
    gt_sig_features <- feature_names[which(z == 1)]
    gt_not_sig_features <- setdiff(feature_names, gt_sig_features)
    
    ret <- list()
    for (method_name in methods)
    {
      results <- eval_method(seurat, method_name, id1, id2)
      ret[[method_name]] <- performance_metrics(results[[1]], results[[2]], gt_sig_features, gt_not_sig_features)
    }
    # Output ret.
    ret_dt <- data.table::rbindlist(ret, idcol = "method")
    ret_dt[,f1 := 2*tpr*prec/(prec + tpr)]
    fwrite(ret_dt, file = paste0(rep_path, "/results.csv"))
  }
}

results_list <- list()
for (dispersion in dispersions)
{
  output_path <- paste0(base_path, "/dispersion_", dispersion)
  dispersion_ <- as.numeric(gsub(pattern = "_", replacement = ".", x = dispersion))
  
  for (rep_no in seq(0, rep_count-1))
  {
    rep_path <- paste0(output_path, "/rep", rep_no)
    ret <- fread(paste0(rep_path, "/results.csv"))
    ret$dispersion <- dispersion_
    ret$rep_no <- rep_no
    results_list[[paste0(dispersion_, "_", rep_no)]] <- ret
  }
}

results_dt <- rbindlist(results_list)
fwrite(results_dt, file = paste0(base_path, "/results.csv"))
pl <- ggplot(results_dt, aes(as.factor(dispersion), prec, fill=method)) + geom_boxplot() + theme_bw() +
  xlab("Dispersion") + ylab("Precision")
ggsave(filename = paste0(base_path, "/precision.pdf"), pl)

pl <- ggplot(results_dt, aes(as.factor(dispersion), tpr, fill=method)) + geom_boxplot() + theme_bw() +
  xlab("Dispersion") + ylab("Recall")
ggsave(filename = paste0(base_path, "/recall.pdf"), pl)

pl <- ggplot(results_dt, aes(as.factor(dispersion), f1, fill=method)) + geom_boxplot() + theme_bw() +
  xlab("Dispersion") + ylab("F1")
ggsave(filename = paste0(base_path, "/f1.pdf"), pl)

pl <- ggplot(results_dt, aes(as.factor(dispersion), acc, fill=method)) + geom_boxplot() + theme_bw() +
  xlab("Dispersion") + ylab("Accuracy")
ggsave(filename = paste0(base_path, "/accuracy.pdf"), pl)
