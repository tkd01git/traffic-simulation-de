import pandas as pd
import matplotlib.pyplot as plt
import os

# スピードの閾値
speed_threshold = 50

# 処理するファイル名のリスト
file_numbers = [8,9,10,11,12,13,14,15,16,17]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
file_names = [f"λrankresult{num}.xlsx" for num in file_numbers]

# 赤線を引く時刻を決定する関数
def identify_red_lines(file_path, sheet_name, speed_threshold):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[['Time', 'speed']].dropna()
    df.columns = ['Time', 'Speed']  # 列名を統一

    thin_red_lines = []  # 赤線を引く時刻を保持するリスト
    for i in range(len(df)):
        if df.iloc[i]['Speed'] < speed_threshold:
            thin_red_lines.append(df.iloc[i]['Time'])  # 薄い赤線の時刻を追加

    return thin_red_lines

# グラフの描画準備
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 各ファイルを処理
for row_idx, file_name in enumerate(file_names):
    file_path = os.path.join(base_path, file_name)
    file_number = int(file_name.replace('λrankresult', '').replace('.xlsx', ''))

    # 速度グラフのプロット
    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_sheet1 = df_sheet1[['Time', 'speed']].dropna()
    ax_speed = axes[row_idx, 0]  # 左側の列
    ax_speed.plot(df_sheet1['Time'], df_sheet1['speed'], label='Speed', color='purple', linewidth=1.2)
    thin_red_lines = identify_red_lines(file_path, 'Sheet1', speed_threshold)

    # 薄い赤線を引く
    for thin_time in thin_red_lines:
        ax_speed.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

    ax_speed.set_title(f'File {file_number} - Speed')
    ax_speed.set_ylabel("Speed (km/h)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.legend()
    ax_speed.grid()

    # Rank of F_lambda2 のプロット
    df_sheet2 = pd.read_excel(file_path, sheet_name='Sheet1')  # 必要に応じてシート名を変更
    if 'Rank of F_lambda2' in df_sheet2.columns:
        df_sheet2 = df_sheet2[['Time', 'Rank of F_lambda2']].dropna().sort_values('Time')
        df_sheet2.columns = ['Time', 'Rank of Frambda2']
    else:
        print(f"File {file_name}: 'Rank of F_lambda2' not found. Skipping.")
        continue

    ax_rank = axes[row_idx, 1]  # 右側の列
    ax_rank.plot(df_sheet2['Time'], df_sheet2['Rank of Frambda2'], color='blue', linewidth=1)

    # 薄い赤線を引く
    for thin_time in thin_red_lines:
        ax_rank.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

    ax_rank.set_ylim(0, 41)
    ax_rank.set_title(f'File {file_number} - Rank of F_lambda2')
    ax_rank.set_ylabel("Rank of F_lambda2")
    ax_rank.set_xlabel("Time (s)")
    ax_rank.grid()

# 全体のタイトルを追加
fig.suptitle("Speed and Rank of F_lambda2", fontsize=16)

# 注意書きを追加
plt.figtext(0.5, 0.01, "Red line = speed below 50 km/h.", ha="center", fontsize=12, color="gray")

plt.show()
