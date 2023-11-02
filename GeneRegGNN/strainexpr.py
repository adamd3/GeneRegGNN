import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .utils import get_expression_cors


class CoexpressNetwork:
    """
    CoexpressNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        feature_ids (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """

    def __init__(self, edge_list, feature_ids, node_map):
        """
        Initialize CoexpressNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(
            self.edge_list,
            source="source",
            target="target",
            edge_attr=["cor_coef"],
            create_using=nx.DiGraph(),
        )
        self.feature_ids = feature_ids
        for n in self.feature_ids:
            if n not in self.G.nodes():
                self.G.add_node(n)

        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        # self.edge_weight = torch.Tensor(self.edge_list['cor_coef'].values)

        edge_attr = nx.get_edge_attributes(self.G, "cor_coef")
        cor_coef = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(cor_coef)


class StrainExpr:
    # TODO: user can pass a data_path to load preprocessed data
    def __init__(self, adata: AnnData, data_path: str):
        """
        Parameters
        ----------

        adata: str
            Annotated gene expression data in h5ad format
        data_path: str
            Path for saving data
        """
        self.adata = ad.read_h5ad(adata)
        self.split = None
        self.data_path = data_path
        self.feature_list = self.adata.var.feature_id.values.tolist()
        self.node_map = {
            x: it for it, x in enumerate(self.adata.var.feature_id)
        }

    def __len__(self):
        return len(self.adata)

    def split_data(self, split="standard", seed=0):
        """
        Split data into train/val/test sets

        Parameters
        ----------

        split: str
            Type of split to use. Options are:
            - no_split: no split, use all data for training
            - no_test: use 80% of data for training, 20% for validation
            - standard: use 80% of data for training, 10% for validation, 10% for testing

        seed: int
            Random seed for reproducibility
        """

        np.random.seed(seed)
        self.split = split

        if split == "no_split":
            self.adata.obs["split"] = "train"
        elif split == "no_test":
            train_idx, val_idx = train_test_split(
                self.adata.obs.index, test_size=0.2, random_state=seed
            )
            self.adata.obs["split"] = "train"
            self.adata.obs.loc[val_idx, "split"] = "validation"
        elif split == "standard":
            train_idx, temp_idx = train_test_split(
                adata.obs.index, test_size=0.2, random_state=seed
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=seed
            )
            self.adata.obs["split"] = "train"
            self.adata.obs.loc[val_idx, "split"] = "validation"
            self.adata.obs.loc[test_idx, "split"] = "test"
        else:
            raise ValueError("Invalid split type")

    # TODO: user can pass a data_path to load preprocessed network + weights
    def initialise_coexpress_network(
        self, k: int = 20, coexpress_threshold: float = 0.4
    ):
        """
        Initialize the coexpression network

        Parameters
        ----------
        k: int
            Number of edges to be retained in the co-expression graph
        coexpress_threshold: float
            Pearson correlation threshold for co-expression graph

        Returns
        -------
        None
        """

        if self.config["G_coexpress"] is None:
            edge_list = get_expression_cors(
                adata=self.adata,
                threshold=coexpress_threshold,
                k=k,
                data_path=self.data_path,
            )

            sim_network = CoexpressNetwork(
                edge_list, self.feature_list, node_map=self.node_map
            )
            self.config["G_coexpress"] = sim_network.edge_index
            self.config["G_coexpress_weight"] = sim_network.edge_weight

    def get_dataloader(
        self, batch_size: int, num_workers: int = 8
    ) -> DataReaderOutput:
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        num_workers: int
            Number of workers for dataloader

        Ret
        -------
        dict
            Dictionary of dataloaders

        """

        if self.split == "no_split":
            test_loader = DataLoader(self.test_dataset, batch_size=batch_size)
            return {"test_loader": test_loader}
        else:
            train_loader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=True
            )

            if self.split != "no_test":
                test_loader = DataLoader(
                    self.test_dataset, batch_size=batch_size, shuffle=False
                )
                self.adataloader = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }

            else:
                self.adataloader = {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                }
