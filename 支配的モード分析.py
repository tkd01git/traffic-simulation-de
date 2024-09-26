import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ファイルパス
file_path = 'C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//sorted_speed_data.xlsx'

# 連続した列を指定
start_col = 'A'
end_col = 'DI'
sheet_name = 'Sheet1'

# Excelファイルを読み込み、連続した列を指定
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=f'{start_col}:{end_col}', header=None)

# シートと列のペアを作成（列のインデックスを取得）
sheet_column_pairs = [(sheet_name, col) for col in range(df.shape[1])]

def read_speed_data(df, column_index):
    """指定された列のデータを読み込みます。"""
    if df.shape[0] > 20:  # 21行以上あるか確認
        speed_data = df.iloc[1:21, column_index].values.flatten()  # 2行目から21行目のデータを取得
        labels = df.iloc[0, column_index]  # 第一行をラベルとして取得
        return speed_data, labels
    else:
        raise ValueError("データフレームに21行以上のデータが必要です。")

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

def compute_graph_fourier_transform(df, sheet_column_pairs, L_normalized, eigenvalues):
    """指定されたシートと列のデータについてグラフフーリエ変換を計算します。"""
    results = {}
    eigenvectors = eigh(L_normalized, eigvals_only=False)[1]
    
    for sheet_name, column_index in sheet_column_pairs:
        speed_data, _ = read_speed_data(df, column_index)

        # 固有値と固有ベクトルに基づくグラフフーリエ変換の計算
        F_lambda = np.zeros(eigenvectors.shape[1], dtype=complex)
        for i in range(eigenvectors.shape[1]):
            u_lambda_i = eigenvectors[:, i]
            F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))
        
        results[(sheet_name, column_index)] = {
            'F_lambda_real': abs(np.real(F_lambda)),
            'eigenvalues': eigenvalues
        }
    
    return results

def plot_eigenvalue_results(results, labels):
    """すべての固有値に基づくデータをプロットします。"""
    num_eigenvalues = len(results[next(iter(results))]['eigenvalues'])
    
    # 4行5列のサブプロットを作成
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    axs = axs.flatten()  # 2次元配列を1次元に変換

    # 縦軸のスケールを統一するための最大値と最小値を計算
    all_F_values = []
    for data in results.values():
        all_F_values.extend(data['F_lambda_real'])
    
    y_min = min(all_F_values)
    y_max = max(all_F_values)

    for i in range(num_eigenvalues):
        for (sheet_name, column_index), data in results.items():
            if len(data['eigenvalues']) >= (i + 1):
                F_value = data['F_lambda_real'][i]
                axs[i].bar(labels[column_index], F_value, color='b', alpha=0.7)
                axs[i].set_title(f'λ_ {i + 1}', fontsize=10)  # 標題のフォントサイズを小さく
                axs[i].set_xlabel('Speed Data (from Excel)', fontsize=8)  # 横軸ラベルのサイズ
                axs[i].set_ylabel(f'F(λ)', fontsize=8)  # 縦軸ラベルのサイズ
                axs[i].set_ylim(y_min, y_max)  # 縦軸のスケールを統一
                axs[i].tick_params(axis='x', rotation=0)  # 横軸の目盛りを水平に
                axs[i].grid(axis='y')
    
    plt.tight_layout()
    plt.show()


def main(file_path, sheet_column_pairs):
    # グラフ構造の設定
    n_nodes = 20
    A = create_adjacency_matrix(n_nodes)
    L, D = compute_laplacian(A)
    L_normalized = compute_normalized_laplacian(L, D)
    eigenvalues, _ = eigh(L_normalized)
    
    # グラフフーリエ変換の計算
    results = compute_graph_fourier_transform(df, sheet_column_pairs, L_normalized, eigenvalues)
    
    # 第一行のラベルを取得
    labels = {}
    for sheet_name, column_index in sheet_column_pairs:
        _, label = read_speed_data(df, column_index)
        labels[column_index] = label

    # すべての固有値に基づくデータをプロット
    plot_eigenvalue_results(results, labels)



# 実行部分
main(file_path, sheet_column_pairs)