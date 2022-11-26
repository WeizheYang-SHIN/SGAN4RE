import numpy as np
import scipy.sparse as sp


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix. 对称归一化邻接矩阵。 """
#     # adj = sp.coo_matrix(adj) # 构造一个矩阵类型的数据变量
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = np.diag(d_inv_sqrt)  # D^(-1/2), D is degree matrix
#
#     '''
#         adj是对称矩阵，因此 adj^T=adj.  .tocoo()
#         d_mat_inv_sqrt是对角矩阵，因此 d_mat_inv_sqrt^T=d_mat_inv_sqrt.
#         (adj*d_mat_inv_sqrt)^T=(d_mat_inv_sqrt^T)*(adj^T) = d_mat_inv_sqrt*adj
#         最后return: d_mat_inv_sqrt*adj*d_mat_inv_sqrt
#     '''
#     ans = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#     return ans

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    '''
        原本的adj的对角线元素为空，这里对角线设置为1，即adj + sp.eye(adj.shape[0])
    '''
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

