
import numpy as np
import pandas as pd
import scipy.sparse as sp


def calculate_normalized_laplacian(adj):
    """
    # D = diag(A 1)
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian.astype(np.float32).todense()


def asym_adj(adj):
    """
    # D = diag(A 1)
    # P = D^-1 A
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(adj_data_path, adj_type, use_graph_learning=False):
    df = pd.read_csv(adj_data_path, header=None)
    adj_mx = df.to_numpy().astype(np.float32)
    if adj_type == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    adaptive_mat = adj[0] if use_graph_learning else None
    return adaptive_mat, adj

def load_multi_adj(path1, path2, use_graph_learning=False):
    df1, df2 = pd.read_csv(path1, header=None), pd.read_csv(path2, header=None)
    adj_mx_geo, adj_mx_flow = df1.to_numpy().astype(np.float32), df2.to_numpy().astype(np.float32)
    adj = [
        asym_adj(adj_mx_geo), asym_adj(adj_mx_flow),
        asym_adj(np.transpose(adj_mx_geo)), asym_adj(np.transpose(adj_mx_flow))
    ]
    adaptive_mat = adj[0] if use_graph_learning else None
    return adaptive_mat, adj


if __name__ == "__main__":
    adj = load_adj('../data/adj_mx_geo_126.csv', 'normlap')
    print(adj)