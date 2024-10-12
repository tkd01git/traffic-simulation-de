import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Excelファイルの読み込みと前処理
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//prodata.xlsx"
start_col = 'HR'
end_col = 'OM'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# 結果を保存するためのDataFrameの作成
results_df = pd.DataFrame(columns=['Time', 'Average Speed'] + [f'Lambda Index No.{i}' for i in range(1, 21)] + [f'|F(λ)| No.{i}' for i in range(1, 21)])

# CSVファイル名
file_name = "002nojam.csv"


# 列名をCSVファイルに追加
results_df.to_csv(file_name, mode='a', header=True, index=False)

# Sheet1とSheet2のデータの取得
df1 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2, usecols=f'{start_col}:{end_col}', header=None)
average_speed_data = df2.iloc[23].values.flatten()

# 列名とデータの前処理
df1.columns = df1.iloc[0]
df1 = df1[1:]

lambdas = np.arange(1, 21)
times = df1.columns.tolist()
all_f_lambda_abs = []
all_top_20_lambdas = []

# Laplacianの作成と計算
n = 20
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = 1
    A[i + 1, i] = 1
L = np.diag(np.sum(A, axis=1)) - A
L_normalized = np.dot(np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))), np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))))

for column_name in df1.columns:
    speed_data = df1[column_name].iloc[0:20].values.flatten()
    eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
    for i in range(sorted_eigenvectors.shape[1]):
        u_lambda_i = sorted_eigenvectors[:, i]
        F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))

    F_lambda_abs_values = np.abs(F_lambda[:20])
    all_f_lambda_abs.append(F_lambda_abs_values)

    top_20_indices = np.argsort(F_lambda_abs_values)[-20:][::-1]
    all_top_20_lambdas.append(top_20_indices + 1)

    row = [column_name, average_speed_data[df1.columns.get_loc(column_name)]]
    row.extend(top_20_indices + 1)
    row.extend(F_lambda_abs_values[top_20_indices])
    results_df.loc[len(results_df)] = row

# データをCSVファイルに追加
results_df.to_csv(file_name, mode='a', header=False, index=False)


print(f"CSV file '{file_name}' has been updated")
