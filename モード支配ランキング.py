import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ファイルパス
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//sorted_speed_data.xlsx"

# 連続した列を指定
start_col = 'A'
end_col = 'DI'
sheet_name = 'Combined_Detectors'

# Excelファイルを読み込み、連続した列を指定
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=f'{start_col}:{end_col}', header=None)

# 列名を取得（1行目に列名が入っていると仮定）
df.columns = df.iloc[0]  # 最初の行を列名として設定
df = df.drop(0)  # 最初の行は削除

# デバッグ: 列名が正しく読み込まれているか確認
print("読み込まれた列名:")
print(df.columns.tolist())  # 列名のリストを表示

def read_speed_data(df, column_name):
    """指定された列のデータを名前で読み込みます。"""
    if df.shape[0] >= 21:  # 21行以上あるか確認
        # 2行目から21行目のデータを取得（ilocで行番号指定）
        speed_data = df[column_name].iloc[0:20].values.flatten()  # 行は0から20行目
        label = column_name  # 列名をラベルとして使用
        return speed_data, label
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

# 固有値計算のための準備
n = 20  # 固有値の数（例として20に設定）
A = create_adjacency_matrix(n)  # 隣接行列の作成
L, D = compute_laplacian(A)  # ラプラシアン行列を計算
L_normalized = compute_normalized_laplacian(L, D)  # 正規化ラプラシアン行列を計算

# 全ての時刻に対する結果を保存するリスト
all_results = []

# 全ての時刻（列）に対して計算を実行
for column_name in df.columns:
    # 各列のデータを取得
    speed_data, _ = read_speed_data(df, column_name)

    # 固有値と固有ベクトルの計算
    eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

    # F(λ)の計算
    F_lambda = np.zeros(eigenvectors.shape[1], dtype=complex)
    for i in range(eigenvectors.shape[1]):
        u_lambda_i = eigenvectors[:, i]
        F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))

    # 結果の表示（絶対値でソートし上位1位を表示）
    results = [(eigenvalues[i], F_lambda[i].real) for i in range(len(F_lambda))]

    # 固有値ゼロのモードを除外
    results = [res for res in results if not np.isclose(res[0], 0)]

    # 固有値を小さい順にソート
    sorted_eigenvalues = sorted(eigenvalues)

    # F(λ)の絶対値でソート
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # 上位1位のデータを保存
    for rank in range(min(1, len(results))):  # 上位1位を保存
        eigenvalue, f_value = results[rank]
        
        # 固有値が小さい順で何番目かを取得
        sorted_rank = sorted_eigenvalues.index(eigenvalue) + 1  # 1から始まるインデックス

        # 結果を保存
        all_results.append([column_name, f"{sorted_rank}", f"{f_value:.4f}"])

# 表を作成するためのデータを準備
table_data = [['time', 'λ index', 'F(λ)']]  # ヘッダー
table_data.extend(all_results)

# グラフの描画
fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.4))  # 表の行数に合わせたサイズ設定

# テーブルを描画
table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

# セルのサイズとフォントサイズを調整
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.5, 1)  # 読みやすさ向上のためスケールを大きくする

# テーブルの枠線を非表示にしない
table.auto_set_column_width([0, 1, 2])

# 軸をオフにして、表だけを表示
ax.axis('tight')
ax.axis('off')

# 表を表示
plt.title('top eigenvalue and F(λ) for each time except for λ=0')
plt.show()
