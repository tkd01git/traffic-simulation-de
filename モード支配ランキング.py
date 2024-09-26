import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ファイルパス
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//sorted_speed_data.xlsx"

# 連続した列を指定
start_col = 'CA'
end_col = 'DI'
sheet_name = 'Sheet1'

# Excelファイルを読み込み、連続した列を指定
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=f'{start_col}:{end_col}', header=None)

# 列名を取得（1行目に列名が入っていると仮定）
df.columns = df.iloc[0]

# デバッグ: 列名が正しく読み込まれているか確認
print("読み込まれた列名:")
print(df.columns.tolist())

def read_speed_data(df, column_name):
    if df.shape[0] >= 21:
        speed_data = df[column_name].iloc[0:20].values.flatten()
        label = column_name
        return speed_data, label
    else:
        raise ValueError("データフレームに21行以上のデータが必要です。")

def create_adjacency_matrix(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A

def compute_laplacian(A):
    row_sums = np.sum(A, axis=1)
    D = np.diag(row_sums)
    L = D - A
    return L, D

def compute_normalized_laplacian(L, D):
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L_normalized = np.dot(D_inv_sqrt, np.dot(L, D_inv_sqrt))
    return L_normalized

n = 20
A = create_adjacency_matrix(n)
L, D = compute_laplacian(A)
L_normalized = compute_normalized_laplacian(L, D)

# 上位結果を格納する辞書
all_results = {}

for column_name in df.columns:
    speed_data, _ = read_speed_data(df, column_name)
    eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

    F_lambda = np.zeros(eigenvectors.shape[1], dtype=complex)
    for i in range(eigenvectors.shape[1]):
        u_lambda_i = eigenvectors[:, i]
        F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))

    results = [(eigenvalues[i], F_lambda[i].real) for i in range(len(F_lambda))]
    results = [res for res in results if not np.isclose(res[0], 0)]
    sorted_eigenvalues = sorted(eigenvalues)
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # 上位3つの結果を保存
    top_results = results[:3]
    if column_name not in all_results:
        all_results[column_name] = []

    for eigenvalue, _ in top_results:
        sorted_rank = sorted_eigenvalues.index(eigenvalue) + 1
        all_results[column_name].append([f"{sorted_rank}"])

# 表を作成するためのデータを準備
table_data = [['time', 'λ index no.1', 'λ index no.2', 'λ index no.3']]
for time, results in all_results.items():
    row = [time]
    for result in results:
        row.extend(result)
    # 上位3つに満たない場合は空白を追加
    while len(row) < 4:
        row.append('')
    table_data.append(row)

# グラフの描画
fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.3))

table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

table.auto_set_column_width([0, 1, 2, 3])
ax.axis('tight')
ax.axis('off')

plt.title('Top 3 Eigenvalues for Each Time Except for λ=0')
plt.show()
