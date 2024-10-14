import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Excelファイルの読み込みと前処理
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//prodata.xlsx"
start_col = 'A'
end_col = 'EY'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# Sheet1とSheet2のデータの取得
df1 = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=f'{start_col}:{end_col}', header=None)
df2 = pd.read_excel(file_path, sheet_name=sheet_name2, usecols=f'{start_col}:{end_col}', header=None)
average_speed_data = df2.iloc[23].values.flatten()  # 平均速度データ

# 列名とデータの前処理
df1.columns = df1.iloc[0]
df1 = df1[1:]

# グラフ描画用のデータ
lambdas = np.arange(1, 21)  # λのインデックスを1~20に修正
times = df1.columns.tolist()  # 時刻（列名）
all_top_20_lambdas = []  # 各時刻ごとの上位20つの固有値インデックスを保存

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

    # 固有値は小さい順に並べ替える
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Fλの計算
    F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
    for i in range(sorted_eigenvectors.shape[1]):
        u_lambda_i = sorted_eigenvectors[:, i]
        F_lambda[i] = np.sum(speed_data * np.conj(u_lambda_i))

    # Fλの絶対値を抽出
    F_lambda_abs_values = np.abs(F_lambda[:20])  # 1~20のλインデックスに対応

    # Fλの絶対値の上位20つを抽出
    top_20_indices = np.argsort(F_lambda_abs_values)[-20:][::-1]  # 絶対値の大きい順に上位20つのインデックスを取得
    all_top_20_lambdas.append(top_20_indices + 1)  # 1から始まるように調整

# 表の描画
table_data = [['time', 'average speed'] + [f'λ index no.{i+1}' for i in range(20)]]
for time, lambdas in zip(df1.columns, all_top_20_lambdas):
    row = [time, average_speed_data[df1.columns.get_loc(time)]]  # 平均速度を追加
    row.extend(lambdas)  # 上位20つのλインデックスを追加
    table_data.append(row)

# 表の色付け
def get_color_for_speed(speed):
    normalized_speed = min(max(speed / 100, 0), 1)  # 0~1に正規化
    return plt.cm.RdYlGn(normalized_speed)  # 赤から緑のグラデーション

def get_color_for_lambda_index(index):
    normalized_index = min(max(index / 30, 0), 1)  # 0~1に正規化
    return plt.cm.Reds(normalized_index)  # 青のグラデーション

# 表の描画
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_data, colLoc='center', loc='center')

# セルの色付け
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#d9d9d9')  # ヘッダー行の色
    elif j == 1:  # 平均速度の列
        speed_value = float(table_data[i][j])
        cell.set_facecolor(get_color_for_speed(speed_value))  # 速度に応じた色付け
    elif 2 <= j <= 21:  # λ indexの列
        lambda_index = int(table_data[i][j])
        cell.set_facecolor(get_color_for_lambda_index(lambda_index))  # λ indexに応じた色付け
    else:
        cell.set_facecolor('white')  # その他のセルは白

table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(0.6, 0.6)

# 表を表示
plt.show()
