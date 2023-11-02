import numpy as np
import pandas as pd
import networkx as nx
import os


def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def get_expression_cors(adata, threshold, k, data_path):
    """
    Get pairwise gene coexpression values from training data

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
    """

    fname = os.path.join(data_path, f"coexpress_network_{threshold}_{k}.csv")

    if os.path.exists(fname):
        return pd.read_csv(fname, sep="\t")

    feature_ids = [f for f in adata.var.feature_id.values]
    idx2gene = dict(zip(range(len(feature_ids)), feature_ids))
    X = adata.X

    X_train = X[adata.obs.split == "train"]

    cors = np_pearson_cor(X_train, X_train)
    cors[np.isnan(cors)] = 0
    cors = np.abs(cors)

    # Sort the indices + values of the cors matrix in descending order along
    # each row + select the top k+1 indices for each gene
    cors_sort_idx = np.argsort(cors)[:, -(k + 1) :]
    cors_sort_val = np.sort(cors)[:, -(k + 1) :]

    pairwise_cors = []
    for i in range(cors_sort_idx.shape[0]):
        target = idx2gene[i]
        for j in range(cors_sort_idx.shape[1]):
            pairwise_cors.append(
                (idx2gene[cors_sort_idx[i, j]], target, cors_sort_val[i, j])
            )
    pairwise_cors = [i for i in pairwise_cors if i[2] > threshold]

    cor_df = pd.DataFrame(pairwise_cors).rename(
        columns={0: "source", 1: "target", 2: "cor_coef"}
    )
    cor_df.to_csv(fname, index=False, sep="\t")

    return cor_df
