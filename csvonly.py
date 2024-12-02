import pandas as pd
import numpy as np
from scipy.linalg import eigh

# Excelファイルの読み込みと前処理
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//prodata.xlsx"
start_col = 'A'
end_col = 'DT'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# Sheet1とSheet2のデータの取得
df1 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2, usecols=f'{start_col}:{end_col}', header=None)

# 新しいデータフレームの作成
df3 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None, skiprows=1, nrows=30)  # 2~31行目
df4 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None, skiprows=4, nrows=30)  # 5~34行目
df5 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None, skiprows=7, nrows=30)  # 8~37行目
df6 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None, skiprows=10, nrows=30) # 11~40行目


# 各時刻の平均速度データと車両数データを取得
time_data = df1.iloc[0].values.flatten()
flow_data = df2.iloc[21].values.flatten() / 3600  # 車両数データの取得
dens_data = df2.iloc[42].values.flatten()

# Laplacianの作成
n = 30
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = 1
    A[i + 1, i] = 1
L = np.diag(np.sum(A, axis=1)) - A
L_normalized = np.dot(np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))), 
                      np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))))

# F(λ_2) を計算する関数
def calculate_F_lambda_2(df):
    results = []
    for column in df.columns:
        speed_data = df[column].values.flatten()
        eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
        for i in range(sorted_eigenvectors.shape[1]):
            u_lambda_i = sorted_eigenvectors[:, i]
            F_lambda[i] = sum(speed_data * np.conj(u_lambda_i))

        # Fλ_2 (インデックス1) の実部を取得
        results.append(F_lambda[1].real)
    return results

# Fλ_2 の計算
frambda2_df3 = calculate_F_lambda_2(df3)
frambda2_df4 = calculate_F_lambda_2(df4)
frambda2_df5 = calculate_F_lambda_2(df5)
frambda2_df6 = calculate_F_lambda_2(df6)

# 出力用のDataFrameを作成
output_data = {
    'Time': list(time_data),
    'flow': list(flow_data),
    'dens': list(dens_data),
    'Flow Sum 19 Points': [''] * len(time_data),  # 空白のまま作成
    'Franbda2_of_df3': frambda2_df3,
    'Franbda2_of_df4': frambda2_df4,
    'Franbda2_of_df5': frambda2_df5,
    'Franbda2_of_df6': frambda2_df6
}

# 長さを調整してDataFrameに変換
max_len = max(len(time_data), len(frambda2_df3), len(frambda2_df4), len(frambda2_df5), len(frambda2_df6))
for key in output_data.keys():
    if len(output_data[key]) < max_len:
        output_data[key].extend([''] * (max_len - len(output_data[key])))

results_df = pd.DataFrame(output_data)

# Excelファイルにデータを書き込む
output_file = "002.xlsx"
with pd.ExcelWriter(output_file, mode='w') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Results')

# Flow Sum 19 Points の計算を行い、Excelファイルを更新
results_df['Flow Sum 19 Points'] = results_df['flow'].rolling(window=19, min_periods=1, center=True).sum()

# Excelファイルに再度書き込む
with pd.ExcelWriter(output_file, mode='w') as writer:
    results_df.to_excel(writer, index=False, sheet_name='Results')

print(f"{output_file}' が保存されました。")
