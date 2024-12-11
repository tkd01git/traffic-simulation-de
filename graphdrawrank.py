import pandas as pd
import matplotlib.pyplot as plt
import os

# 各ファイルとシートを処理
speed_threshold = 50  # スピードの閾値


# 処理するファイル名のリスト
file_numbers = [16,17,18,19,20]
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


# シート名とタイトル
sheet_titles = {
    'Sheet1': '1',
    'Sheet2': '2',
    'Sheet3': '3',
    'Sheet4': '4',
    'Sheet5': '5',
    'Sheet6': '6',
    'Sheet7': '7',
    'Sheet8': '8'
}


# グラフの描画準備
fig, axes = plt.subplots(nrows=9, ncols=5, figsize=(25, 20), sharex=True, sharey=False)
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 各ファイルを処理
for col_idx, file_name in enumerate(file_names):
    file_path = os.path.join(base_path, file_name)
    # ファイル番号を抽出
    file_number = int(file_name.replace('λrankresult', '').replace('.xlsx', ''))

    # Sheet1 の速度をプロット（1行目）
    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_sheet1 = df_sheet1[['Time', 'speed']].dropna()
    ax = axes[0, col_idx]  # 1列目の最上段
    ax.plot(df_sheet1['Time'], df_sheet1['speed'], label='Speed', color='purple', linewidth=1.2)
    thin_red_lines = identify_red_lines(file_path, 'Sheet1', speed_threshold)

    # 薄い赤線を引く
    for thin_time in thin_red_lines:
        ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

    ax.set_title(f'File {file_number} - Speed')
    ax.legend()
    ax.grid()

    # Sheet2~Sheet8 の F_lambda2 をプロット（2~9行目）
    for row_idx, (sheet, title) in enumerate(sheet_titles.items(), start=1):  # row_idx を 1 から開始
        thin_red_lines = identify_red_lines(file_path, sheet, speed_threshold)
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df[['Time', 'Rank of F_lambda2']].dropna().sort_values('Time')
        df.columns = ['Time', 'Rank of Frambda2']

        ax = axes[row_idx, col_idx]  # 適切な行に配置
        ax.plot(df['Time'], df['Rank of Frambda2'], color='blue', linewidth=1)

        # 薄い赤線を引く
        for thin_time in thin_red_lines:
            ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        ax.set_ylim(0, 20)
        if col_idx == 0:
            ax.set_ylabel(f'Part {title}', fontsize=8)
        ax.grid()
        ax.axhline(y=0, color='black', linewidth=2)
    
# 全体のタイトルを追加
fig.suptitle("4s-500veh/h : rank of F_lambda2", fontsize=16)

# 注意書きを追加
plt.figtext(0.5, 0.01, "Red line = speed below 50 km/h.", ha="center", fontsize=12, color="gray")

plt.show()
