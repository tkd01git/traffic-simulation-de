
import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def read_data(file_path, sheet_name, column_name):
    """指定した列データをExcelファイルから読み込みます。"""
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=column_name, header=None, skiprows=1, nrows=20)
        return data.values.flatten()
    except Exception as e:
        return None

def create_adjacency_matrix(n):
    """単純な線形グラフの隣接行列を作成します。"""
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
    """ラプラシアン行列を正規化します。"""
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    return L_normalized

def approximate_zero(value, threshold=1e-10):
    """指定した閾値に基づいてゼロに近い値を近似します。"""
    return np.where(np.abs(value) < threshold, 0.0, value)

def compute_graph_fourier_transform(file_path, sheet_column_pairs, eigenvalue_range, L_normalized, eigenvalues):
    """指定したシートと列についてグラフフーリエ変換を計算します。"""
    results = {}
    
    # 固有値と固有ベクトルの計算
    eigenvectors = eigh(L_normalized, eigvals_only=False)[1]
    
    for sheet_name, column_name in sheet_column_pairs:
        speed_data = read_data(file_path, sheet_name, column_name)

        if speed_data is None:
            continue
        
        # 指定された範囲に基づいて固有値と固有ベクトルをフィルタリング
        mask = (eigenvalues >= eigenvalue_range[0]) & (eigenvalues <= eigenvalue_range[1])
        filtered_eigenvalues = eigenvalues[mask]
        filtered_eigenvectors = eigenvectors[:, mask]
        
        if filtered_eigenvalues.size > 0:
            # グラフフーリエ変換の計算
            F_lambda = np.array([np.sum(speed_data * np.conj(filtered_eigenvectors[:, i])) for i in range(filtered_eigenvectors.shape[1])])
            results[(sheet_name, column_name)] = {
                'eigenvalues': filtered_eigenvalues,
                'F_lambda_real': np.abs(np.real(F_lambda))
            }
    
    return results

def plot_results(results):
    """グラフフーリエ変換の結果をプロットします。"""
    plt.figure(figsize=(12, 8))
    
    for (sheet_name, column_name), data in results.items():
        plt.plot(data['eigenvalues'], data['F_lambda_real'], 'o-', label=f'Sheet: {sheet_name}, Column: {column_name}')
    
    plt.xlabel('Eigenvalue λ')
    plt.ylabel('F(λ)')
    plt.title('Graph Fourier Transform Results')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(file_path, sheet_column_pairs, eigenvalue_range=(None, None)):
    # グラフ構造の設定
    n = 20
    A = create_adjacency_matrix(n)
    L, D = compute_laplacian(A)
    L_normalized = compute_normalized_laplacian(L, D)
    eigenvalues, _ = eigh(L_normalized)
    
    # グラフフーリエ変換の計算
    results = compute_graph_fourier_transform(file_path, sheet_column_pairs, eigenvalue_range, L_normalized, eigenvalues)
    plot_results(results)

# 実行部分
if __name__ == "__main__":
    file_path = 'C://Users//YuheiTakada//Downloads//0922simulation.xlsx'
    sheet_column_pairs = [(f'Sheet1', col) for col in ['AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 
                                                         'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 
                                                         'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 
                                                         'BE', 'BF', 'BG', 'BH', 'BI', 'BJ']]
    eigenvalue_range = (1.4, 1.6)  # 固有値の範囲を指定
    main(file_path, sheet_column_pairs, eigenvalue_range)
