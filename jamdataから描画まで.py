import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os

# Laplacian行列の作成
def create_laplacian(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    L = np.diag(np.sum(A, axis=1)) - A
    return np.dot(
        np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))),
        np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))))
    )

# Fλの実部を計算する関数
def calculate_F_lambda_real_parts(df, L_normalized):
    real_parts_list = []
    for column in df.columns:
        speed_data = df[column].values.flatten()
        if len(speed_data) != 40:
            raise ValueError("Speed data length does not match Laplacian matrix dimensions (40).")

        eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
        for i in range(sorted_eigenvectors.shape[1]):
            u_lambda_i = sorted_eigenvectors[:, i]
            F_lambda[i] = sum(speed_data * np.conj(u_lambda_i))

        real_parts = F_lambda.real
        real_parts_list.append(real_parts)
    return real_parts_list

# 赤線を引く時刻を決定する関数
def identify_red_lines(df, speed_threshold):
    df = df[['Time', 'speed']].dropna()
    df.columns = ['Time', 'Speed']
    thin_red_lines = []
    for i in range(len(df)):
        if df.iloc[i]['Speed'] < speed_threshold:
            thin_red_lines.append(df.iloc[i]['Time'])
    return thin_red_lines

# ファイルの処理
def process_files(file_numbers, base_path, speed_threshold):
    # Laplacian行列の準備
    n = 40
    L_normalized = create_laplacian(n)

    # グラフの描画準備
    fig, axes = plt.subplots(nrows=len(file_numbers), ncols=2, figsize=(15, 5 * len(file_numbers)), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for row_idx, file_number in enumerate(file_numbers):
        file_path = os.path.join(base_path, f"jamdata{file_number}.xlsx")

        # ファイル読み込み
        df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None, skiprows=1, nrows=40)
        df_sheet1.columns = [f"Detector_{i+1}" for i in range(df_sheet1.shape[1])]

        # Laplacian行列を用いてFλの計算
        real_parts_list = calculate_F_lambda_real_parts(df_sheet1, L_normalized)

        # 時間と速度データを取得
        time_data = range(1, 41)
        speed_data = df_sheet1.iloc[:, 0]  # 最初のカラムを速度データと仮定

        # 赤線の時刻を特定
        thin_red_lines = identify_red_lines(pd.DataFrame({'Time': time_data, 'speed': speed_data}), speed_threshold)

        # 速度グラフを描画
        ax_speed = axes[row_idx, 0]
        ax_speed.plot(time_data, speed_data, label='Speed', color='purple', linewidth=1.2)
        for thin_time in thin_red_lines:
            ax_speed.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
        ax_speed.set_title(f'File {file_number} - Speed', fontsize=10)
        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.set_xlabel("Time (s)")
        ax_speed.legend()
        ax_speed.grid()
        ax_speed.set_ylim(0, 100)

        # F_lambda2のグラフを描画
        ax_frambda = axes[row_idx, 1]
        if len(real_parts_list) > 1:
            ax_frambda.plot(time_data, [real_parts[1] for real_parts in real_parts_list], color='blue', linewidth=1)
            for thin_time in thin_red_lines:
                ax_frambda.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
        ax_frambda.set_title(f'File {file_number} - F_lambda2', fontsize=10)
        ax_frambda.set_ylabel("F_lambda2")
        ax_frambda.set_xlabel("Time (s)")
        ax_frambda.grid()
        ax_frambda.set_ylim(-120, 120)

    # 全体のタイトルを追加
    fig.suptitle("Speed and F_lambda2", fontsize=16)
    plt.figtext(0.5, 0.01, "Red line = speed below 50 km/h.", ha="center", fontsize=12, color="gray")

    plt.show()

# 実行設定
if __name__ == "__main__":
    file_numbers = [1, 2, 3, 6, 5]  # 処理するファイル番号
    base_path = "./"  # ファイルが存在するディレクトリ
    speed_threshold = 50
    process_files(file_numbers, base_path, speed_threshold)
