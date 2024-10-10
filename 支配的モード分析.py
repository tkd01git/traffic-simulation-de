import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Excelファイルの読み込みと前処理
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//prodata.xlsx"
start_col = 'E'
end_col = 'PP'
sheet_name1 = 'Sheet1'
sheet_name2 = 'Sheet2'

# 3Dプロット用の関数（棒グラフ）
def plot_3d_bar_with_top3(lambdas, times, F_lambda_abs_values, top3_indices_per_time):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection='3d')

    X, Y = np.meshgrid(lambdas, times)
    Z = np.array(F_lambda_abs_values)
    xpos, ypos = X.flatten(), Y.flatten()
    zpos = np.zeros_like(xpos)

    # 棒の幅と深さ
    dx = dy = 0.7  # 少し狭めに調整して軽量化
    dz = Z.flatten()
    

    # 各時刻の上位3つのFλの値に特別な色を付ける
    colors = np.full_like(dz, fill_value='gray', dtype=object)  # 全ての棒は初期状態で灰色に変更
    for t_idx, top3 in enumerate(top3_indices_per_time):
        for top_idx in top3:
            lambda_pos = lambdas[top_idx - 1] - 1  # 1始まりなので調整
            bar_index = t_idx * len(lambdas) + lambda_pos
            colors[bar_index] = 'r'  # 上位3つのFλは赤に設定

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    ax.set_xlabel('Lambda Index')
    ax.set_ylabel('Time')
    ax.set_zlabel('|F(λ)|')
    ax.set_title('3D Bar Plot of |F(λ)| by Time and Lambda Index with Top 3 Highlighted')
    
    return fig, ax



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
all_f_lambda_abs = []
all_top_3_lambdas = []  # 各時刻ごとの上位3つの固有値インデックスを保存

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
    all_f_lambda_abs.append(F_lambda_abs_values)

    # Fλの絶対値の上位3つを抽出
    top_3_indices = np.argsort(F_lambda_abs_values)[-3:][::-1]  # 絶対値の大きい順に上位3つのインデックスを取得
    all_top_3_lambdas.append(top_3_indices + 1)  # 1から始まるように調整

# 3D棒グラフを描画（top3は赤色で表示、残りは灰色）
fig, ax = plot_3d_bar_with_top3(lambdas, times, all_f_lambda_abs, all_top_3_lambdas)

# 表の描画
table_data = [['time', 'avespeed', 'λ index no.1', 'λ index no.2', 'λ index no.3', '|F(λ)| no.1', '|F(λ)| no.2', '|F(λ)| no.3']]
for time, results, lambdas in zip(df1.columns, all_f_lambda_abs, all_top_3_lambdas):
    row = [time, average_speed_data[df1.columns.get_loc(time)]]  # 平均速度を追加
    
    # λ indexを先に追加
    row.extend(lambdas)  # 上位3つのλインデックスを追加
    
    # 対応するFλの絶対値を追加
    for f_abs_value in results[lambdas - 1]:  # インデックスに対応するFλ絶対値を追加
        row.append(f"{f_abs_value:.2f}")
    
    table_data.append(row)

# 表の色付け
def get_color_for_speed(speed):
    normalized_speed = min(max(speed / 100, 0), 1)  # 0~1に正規化
    return plt.cm.RdYlGn(normalized_speed)  # 赤から緑のグラデーション

def get_color_for_lambda_index(index):
    normalized_index = min(max(index / 30, 0), 1)  # 0~1に正規化
    return plt.cm.Reds(normalized_index)  # 青のグラデーション

def get_color_for_f_lambda(f_lambda_value):
    normalized_value = min(max(f_lambda_value / 500, 0), 1)  # 0~1に正規化
    return plt.cm.Reds(normalized_value)  # 白から赤のグラデーション

# 表の描画
ax_table = fig.add_subplot(122)
table = ax_table.table(cellText=table_data, colLoc='center', loc='center')

# セルの色付け
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#d9d9d9')  # ヘッダー行の色
    elif j == 1:  # 平均速度の列
        speed_value = float(table_data[i][j])
        cell.set_facecolor(get_color_for_speed(speed_value))  # 速度に応じた色付け
    elif 2 <= j <= 4:  # λ indexの列
        lambda_index = int(table_data[i][j])
        cell.set_facecolor(get_color_for_lambda_index(lambda_index))  # λ indexに応じた色付け
    elif 5 <= j <= 7:  # Fλの絶対値の列
        f_lambda_value = float(table_data[i][j])
        cell.set_facecolor(get_color_for_f_lambda(f_lambda_value))  # Fλの絶対値に応じた色付け
    else:
        cell.set_facecolor('white')  # その他のセルは白

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
ax_table.axis('off')

# グラフと表を表示
plt.show()
