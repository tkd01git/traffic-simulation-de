import pandas as pd
import numpy as np
from scipy.linalg import eigh

# Excelファイルの読み込みと前処理
file_path = "C://Users//Ytakada//Downloads//traffic-simulation-de//prodata.xlsx"
start_col = 'A'
end_col = 'DY'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# CSVファイル名
file_name = "004.csv"

# Sheet1とSheet2のデータの取得
df1 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2, usecols=f'{start_col}:{end_col}', header=None)

# 各時刻の平均速度データと車両数データを取得
time_data = df1.iloc[0].values.flatten()
average_speed_data = df2.iloc[44].values.flatten()
avespeed1_20 = df2.iloc[47].values.flatten()
avespeed22_41 = df2.iloc[50].values.flatten()
flow_data = df2.iloc[21].values.flatten() / 3600  # 車両数データの取得
dens_data = df2.iloc[51].values.flatten()

# 列名を指定の順に設定
results_df = pd.DataFrame(columns=[
    'Time', 'Average Speed', 
    'Avespeed1_10', 'Avespeed11_20',
    'flow', 'dens',
    'Flow Sum 19 Points',  # 新しい列を追加
    *[f'F(λ_{i})' for i in range(1, 42)]
])

# Laplacianの作成と計算
n = 41
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = 1
    A[i + 1, i] = 1
L = np.diag(np.sum(A, axis=1)) - A
L_normalized = np.dot(np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))), 
                      np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))))

# 各列ごとにF(λ)を計算
for column_name in df1.columns:
    speed_data = df1[column_name].iloc[1:42].values.flatten()
    eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
    for i in range(sorted_eigenvectors.shape[1]):
        u_lambda_i = sorted_eigenvectors[:, i]
        F_lambda[i] = sum(speed_data * np.conj(u_lambda_i))  # 絶対値を取らずに計算

    # データフレームに書き込む行を作成
    col_index = df1.columns.get_loc(column_name)
    row = [
        time_data[col_index],  # Time
        average_speed_data[col_index],  # Average Speed
        avespeed1_20[col_index],  # Avespeed1~20
        avespeed22_41[col_index],  # Avespeed22~41
        flow_data[col_index],  # carnum
        dens_data[col_index],  # dens
        0,  # Flow Sum 19 Points は後で計算するため一旦0を挿入
        *[F_lambda[i].real for i in range(41)]  # 1~41のF(λ)をすべて出力
    ]
    results_df.loc[len(results_df)] = row

# Flow Sum 19 Points の計算を後から行う
for i in range(len(results_df)):
    start_index = max(0, i - 8)
    end_index = min(len(results_df), i + 9)
    flow_sum_19_points = sum(results_df.loc[start_index:end_index, 'flow'])
    results_df.at[i, 'Flow Sum 19 Points'] = flow_sum_19_points

# CSVファイルにデータを追記
results_df.to_csv(file_name, mode='w', header=True, index=False)

print(f"CSV file '{file_name}' has been updated.")