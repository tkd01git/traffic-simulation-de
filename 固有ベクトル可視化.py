import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh

def create_adjacency_matrix(n):
    """グラフの隣接行列を作成します。"""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A

def compute_laplacian(A):
    """隣接行列からラプラシアン行列を計算します。"""
    row_sums = np.sum(A, axis=1)
    D = np.diag(row_sums)
    L = D - A
    return L, D

def compute_normalized_laplacian(L, D):
    """ラプラシアン行列の正規化を行います。"""
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L_normalized = np.dot(D_inv_sqrt, np.dot(L, D_inv_sqrt))
    return L_normalized

# ノード数
N = 20

# 隣接行列を作成し、ラプラシアン行列と正規化ラプラシアンを計算
A = create_adjacency_matrix(N)
L, D = compute_laplacian(A)
L_normalized = compute_normalized_laplacian(L, D)

# ラプラシアンの固有値と固有ベクトルを計算
eigenvalues, eigenvectors = eigh(L_normalized)

# グラフのレイアウトを直線上に設定
pos = {i: (i, 0) for i in range(N)}

# 描画設定
fig, axes = plt.subplots(4, 5, figsize=(20, 15))  # 4行5列のサブプロット

for idx in range(N):
    ax = axes[idx // 5, idx % 5]  # 4x5のサブプロットの位置指定

    # 現在の固有ベクトル
    fiedler_vector = eigenvectors[:, idx]

    # ノードの描画
    nx.draw(nx.path_graph(N), pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, ax=ax)

    # 固有ベクトルの各成分をノードに対応させて可視化
    for i in range(N):
        # ノード位置
        x, y = pos[i]
        # 信号の強さに応じて矢印を描画（上：正、下：負）
        ax.arrow(x, y, 0, fiedler_vector[i], head_width=0.1, head_length=0.1, fc='r', ec='r')

    # ラベルをつける
    ax.set_title(f'Eigenvector {idx+1}')

# 全体のラベルを設定
plt.suptitle('Laplacian Eigenvectors Visualization (N=20)', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # タイトルが重ならないように調整
plt.show()
