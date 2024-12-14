import pandas as pd
import matplotlib.pyplot as plt
import os

# スピードの閾値
speed_threshold = 50

# 処理するファイル名のリスト
file_numbers = [8,9,10,11,12,13,14,15,16,17]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
file_names = [f"result{num}.xlsx" for num in file_numbers]

# 赤線を引く時刻を決定する関数
def identify_red_lines(df, speed_threshold):
    return df.loc[df['Speed'] < speed_threshold, 'Time'].tolist()

# グラフの描画準備
fig, axes = plt.subplots(nrows=len(file_numbers), ncols=2, figsize=(10, 12), sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.3, wspace=0.2)  # グラフ間の余白を調整

# 各ファイルを処理
for row_idx, file_name in enumerate(file_names):
    file_path = os.path.join(base_path, file_name)
    file_number = int(file_name.replace('result', '').replace('.xlsx', ''))

    try:
        # ファイルとシート1を読み込み
        df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
        df_sheet1 = df_sheet1[['Time', 'speed', 'F_lambda2']].dropna()
        df_sheet1.columns = ['Time', 'Speed', 'Frambda2']  # 列名を統一

        # 赤線を引く時刻を決定
        thin_red_lines = identify_red_lines(df_sheet1, speed_threshold)

        # 速度グラフのプロット
        ax_speed = axes[row_idx, 0]  # 左側の列
        ax_speed.plot(df_sheet1['Time'], df_sheet1['Speed'], label='Speed', color='purple', linewidth=1.2)

        # 薄い赤線を引く
        for thin_time in thin_red_lines:
            ax_speed.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        ax_speed.set_title(f'File {file_number} - Speed', fontsize=9)
        ax_speed.set_ylabel("Speed (km/h)", fontsize=8)
        ax_speed.set_xlabel("Time (s)", fontsize=8)
        ax_speed.legend(fontsize=8)
        ax_speed.grid()
        ax_speed.set_ylim(0, 100)

        # F_lambda2 のプロット
        ax_frambda = axes[row_idx, 1]  # 右側の列
        ax_frambda.plot(df_sheet1['Time'], df_sheet1['Frambda2'], color='blue', linewidth=1)

        # 薄い赤線を引く
        for thin_time in thin_red_lines:
            ax_frambda.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        ax_frambda.set_ylim(-120, 120)
        ax_frambda.set_title(f'File {file_number} - F_lambda2', fontsize=9)
        ax_frambda.set_ylabel("F_lambda2", fontsize=8)
        ax_frambda.set_xlabel("Time (s)", fontsize=8)
        ax_frambda.grid()

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        continue

# 全体のタイトルを追加
fig.suptitle("Speed and F_lambda2", fontsize=14)

# 注意書きを追加
plt.figtext(0.5, 0.01, "Red line = speed below 50 km/h.", ha="center", fontsize=10, color="gray")

plt.show()
