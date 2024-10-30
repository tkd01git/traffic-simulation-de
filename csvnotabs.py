import pandas as pd
import numpy as np
from scipy.linalg import eigh

# Excelファイルの読み込みと前処理
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//prodata.xlsx"
start_col = 'A'
end_col = 'NJ'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# CSVファイル名
file_name = "012.csv"

# Sheet1とSheet2のデータの取得
df1 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2, usecols=f'{start_col}:{end_col}', header=None)

# 各時刻の平均速度データを取得
average_speed_data = df2.iloc[23].values.flatten()
avespeed1_10 = df2.iloc[26].values.flatten()
avespeed11_20 = df2.iloc[29].values.flatten()
avespeed1_5 = df2.iloc[32].values.flatten()
avespeed6_15 = df2.iloc[35].values.flatten()
avespeed16_20 = df2.iloc[38].values.flatten()

# 列名を指定の順に設定
results_df = pd.DataFrame(columns=[
    'Time', 'Average Speed', 
    'avespeed1_10', 'avespeed11_20', 
    'avespeed1_5', 'avespeed6_15', 
    'avespeed16_20', 
    *[f'F(λ_{i})' for i in range(1, 4)]
])

# Laplacianの作成と計算
n = 20
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = 1
    A[i + 1, i] = 1
L = np.diag(np.sum(A, axis=1)) - A
L_normalized = np.dot(np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))), 
                      np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))))

# 各列ごとにF(λ)を計算
for column_name in df1.columns:
    speed_data = df1[column_name].iloc[0:20].values.flatten()
    eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
    for i in range(sorted_eigenvectors.shape[1]):
        u_lambda_i = sorted_eigenvectors[:, i]
        F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))

    # データフレームに書き込む行を作成
    row = [
        column_name,  # Time
        average_speed_data[df1.columns.get_loc(column_name)],  # Average Speed
        avespeed1_10[df1.columns.get_loc(column_name)].tolist(),  # Avespeed1~10
        avespeed11_20[df1.columns.get_loc(column_name)].tolist(),  # Avespeed11~20
        avespeed1_5[df1.columns.get_loc(column_name)].tolist(),  # Avespeed1~5
        avespeed6_15[df1.columns.get_loc(column_name)].tolist(),  # Avespeed6~15
        avespeed16_20[df1.columns.get_loc(column_name)].tolist(),  # Avespeed16~20
        *F_lambda[:3].real  # F(λ_1)~F(λ_20)
    ]
    results_df.loc[len(results_df)] = row

# CSVファイルにデータを追記
results_df.to_csv(file_name, mode='a', header=True, index=False)

print(f"CSV file '{file_name}' has been updated.")
