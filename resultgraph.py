import pandas as pd
import matplotlib.pyplot as plt
import os

# スピードの閾値
speed_threshold = 50

# 処理するファイル名のリスト
file_numbers = [1]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
file_names = [f"result{num}.xlsx" for num in file_numbers]
rank_file_names = [f"λrankresult{num}.xlsx" for num in file_numbers]

# 赤線を引く時刻を決定する関数
def identify_red_lines(df, speed_threshold):
    return df.loc[df['Speed'] < speed_threshold, 'Time'].tolist()

# グラフの描画準備
num_files = len(file_numbers)
fig, axes = plt.subplots(nrows=num_files, ncols=3, figsize=(15, 5 * num_files), sharex=False, sharey=False)

# 1つのファイルしかない場合、axes をリストとして扱う
if num_files == 1:
    axes = [axes]  # サブプロットをリスト化
    
# F_lambda2 が連続して負の値を取る時刻を取得する関数
def identify_yellow_lines(df, consecutive_threshold=5):
    """連続して負の値を取る時刻を特定"""
    negative_indices = df['F_lambda2'] < 0
    groups = negative_indices.ne(negative_indices.shift()).cumsum()
    negative_streaks = df.groupby(groups).filter(lambda g: len(g) >= consecutive_threshold and g['F_lambda2'].iloc[0] < 0)
    return negative_streaks['Time'].tolist()

# 各ファイルを処理
for row_idx, (file_name, rank_file_name) in enumerate(zip(file_names, rank_file_names)):
    file_path = os.path.join(base_path, file_name)
    rank_file_path = os.path.join(base_path, rank_file_name)
    file_number = int(file_name.replace('result', '').replace('.xlsx', ''))

    try:
        # ファイルとシート1を読み込み
        df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
        df_sheet1 = df_sheet1[['Time', 'speed', 'F_lambda2']].dropna()
        df_sheet1.columns = ['Time', 'Speed', 'F_lambda2']  # 列名を統一

        # 赤線を引く時刻を決定
        thin_red_lines = identify_red_lines(df_sheet1, speed_threshold)

        # 黄色線を引く時刻を決定（連続する負の値を考慮）
        yellow_lines = identify_yellow_lines(df_sheet1, consecutive_threshold=10)

        # Rank of F_lambda2 を読み込み
        df_sheet2 = pd.read_excel(rank_file_path, sheet_name='Sheet1')
        if 'Rank of F_lambda2' in df_sheet2.columns:
            df_sheet2 = df_sheet2[['Time', 'Rank of F_lambda2']].dropna()
        else:
            print(f"File {rank_file_name}: 'Rank of F_lambda2' not found. Skipping.")
            continue

        # 速度グラフのプロット
        ax_speed = axes[row_idx][0]  # 左側の列
        ax_speed.plot(df_sheet1['Time'], df_sheet1['Speed'], label='Speed', color='purple', linewidth=1.2)

        for thin_time in thin_red_lines:  # 薄い赤線を描画
            ax_speed.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        ax_speed.set_title(f'File {file_number} - Speed', fontsize=9)
        ax_speed.set_ylabel("Speed (km/h)", fontsize=8)
        ax_speed.set_xlabel("Time (s)", fontsize=8)
        ax_speed.legend(fontsize=8)
        ax_speed.grid()
        ax_speed.set_ylim(0, 100)

        # F_lambda2 グラフのプロット
        ax_frambda = axes[row_idx][1]  # 中央の列
        ax_frambda.plot(df_sheet1['Time'], df_sheet1['F_lambda2'], color='blue', linewidth=1)

        for thin_time in thin_red_lines:
            ax_frambda.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        for yellow_time in yellow_lines:  # 負の値に対応する連続時刻で黄色線を描画
            ax_frambda.axvline(x=yellow_time, color='yellow', linestyle='--', alpha=0.8, linewidth=1)

        ax_frambda.set_ylim(-120, 120)
        ax_frambda.set_title(f'File {file_number} - F_lambda2', fontsize=9)
        ax_frambda.set_ylabel("F_lambda2", fontsize=8)
        ax_frambda.set_xlabel("Time (s)", fontsize=8)
        ax_frambda.grid()

        # Rank of F_lambda2 グラフのプロット
        ax_rank = axes[row_idx][2]  # 右側の列
        ax_rank.plot(df_sheet2['Time'], df_sheet2['Rank of F_lambda2'], color='green', linewidth=1)

        for thin_time in thin_red_lines:
            ax_rank.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        ax_rank.set_ylim(0, 40)
        ax_rank.set_title(f'File {file_number} - Rank of F_lambda2', fontsize=9)
        ax_rank.set_ylabel("Rank of F_lambda2", fontsize=8)
        ax_rank.set_xlabel("Time (s)", fontsize=8)
        ax_rank.grid()

    except Exception as e:
        print(f"Error processing file {file_name} or {rank_file_name}: {e}")
        continue

# 全体のタイトルを追加
fig.suptitle("top40mode: Speed, F_lambda2, and Rank of F_lambda2", fontsize=14)

# 注意書きを追加
plt.figtext(0.5, 0.01, "Red line = speed below 40 km/h. Yellow dashed line = F_lambda2 is negative for 5+ consecutive times.", ha="center", fontsize=10, color="gray")

plt.show()
