import pandas as pd
import matplotlib.pyplot as plt
import os

# 処理するファイル名のリストを作成
file_numbers = [500, 600, 700, 800, 900]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
file_names = [f"allFrambda20-2s_{num}.xlsx" for num in file_numbers]

# 処理するシート名と対応するグラフタイトル
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

# 各ファイルごとに赤線を引く特定の時刻を指定
red_lines_dict = {
    500: [],  # ファイル 500 用の赤線時刻
    600: [],  # ファイル 600 用の赤線時刻
    700: [],  # ファイル 700 用の赤線時刻
    800: [351],  # ファイル 800 用の赤線時刻
    900: [332]   # ファイル 900 用の赤線時刻
}

# 各ファイルごとに青線を引く特定の時刻を指定
blue_lines_dict = {
    500: [],  # ファイル 500 用の青線時刻
    600: [],  # ファイル 600 用の青線時刻
    700: [],  # ファイル 700 用の青線時刻
    800: [],  # ファイル 800 用の青線時刻
    900: []   # ファイル 900 用の青線時刻
}

# グラフの描画準備
fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(25, 20), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 全体タイトルを追加
fig.suptitle('1sdata : F_lambda2 for onramp = 500~900', fontsize=18, y=0.95)

# 凡例の説明を追加
legend_text = (
    "Red Line: Congestion starts\n"
    "Blue Line: Congestion resolves\n"
    "Yellow Line: |Frambda2| ≥ 40"
)
plt.figtext(0.5, 0.01, legend_text, wrap=True, horizontalalignment='center', fontsize=12, color='black')

# 各ファイルとシートを処理
for col_idx, file_name in enumerate(file_names):
    file_path = os.path.join(base_path, file_name)
    file_number = int(file_name.split('_')[-1].replace('.xlsx', ''))  # ファイル番号を取得
    red_lines = red_lines_dict[file_number]  # 該当ファイルの赤線時刻を取得
    blue_lines = blue_lines_dict[file_number]  # 該当ファイルの青線時刻を取得

    for row_idx, (sheet, title) in enumerate(sheet_titles.items()):
        # データを読み込み
        df = pd.read_excel(file_path, sheet_name=sheet)
        
        # 必要な列のみに絞る
        df = df[['Time', 'F_lambda2']]  # 列名は実際のデータに合わせて調整
        df.columns = ['Time', 'Frambda2']  # 列名をわかりやすくする
        df = df.dropna().sort_values('Time')  # 欠損値を削除してソート

        # Frambda2の絶対値が40を超える時刻を特定
        yellow_lines = df.loc[df['Frambda2'].abs() >= 40, 'Time'].tolist()

        # グラフ描画
        ax = axes[row_idx, col_idx]  # 行・列インデックスで指定
        ax.plot(df['Time'], df['Frambda2'], color='blue', linewidth=1)
        
        # 特定の時刻に赤線を引く
        for red_time in red_lines:
            ax.axvline(x=red_time, color='red', linestyle='--', linewidth=1.2)

        # 特定の時刻に青線を引く
        for blue_time in blue_lines:
            ax.axvline(x=blue_time, color='blue', linestyle='--', linewidth=1.2)

        # Frambda2の絶対値が40以上の場合に黄色の縦線を引く
        for yellow_time in yellow_lines:
            ax.axvline(x=yellow_time, color='yellow', linestyle='--', linewidth=1.2)

        # Y軸のスケールを統一
        ax.set_ylim(-60, 60)

        # グラフ設定
        if col_idx == 0:  # 最左列のみにタイトルを表示
            ax.set_ylabel(f'Sheet {title}', fontsize=8)
        if row_idx == 7:  # 最下行のみにX軸ラベルを表示
            ax.set_xlabel(f'onramp = {file_number} [veh/h]', fontsize=8)
        ax.grid()

# グラフ表示
plt.show()
