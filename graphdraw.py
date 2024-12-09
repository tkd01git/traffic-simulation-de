import pandas as pd
import matplotlib.pyplot as plt
import os

# 赤線を引く時刻を決定する関数
def identify_red_lines(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[['Time', 'Density', 'traffic volume']]  # 必要な列を選択
    df.columns = ['Time', 'Density', 'traffic volume']  # 列名を変更
    df = df.dropna()  # 欠損値を削除
    
    thin_red_lines = []  # 薄い赤線を引く時刻を保持するリスト
    thick_red_lines = []  # 濃い赤線を引く時刻を保持するリスト
    
    for i in range(1, len(df)):
        prev_density = df.iloc[i - 1]['Density']
        prev_flow = df.iloc[i - 1]['traffic volume']
        current_density = df.iloc[i]['Density']
        current_flow = df.iloc[i]['traffic volume']
        
        if current_density > prev_density and current_flow < prev_flow:
            thin_red_lines.append(df.iloc[i]['Time'])  # 薄い赤線の時刻を追加
            
    return thin_red_lines, thick_red_lines

# 処理するファイル名のリスト
file_numbers = [1, 2, 3, 4, 5]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
file_names = [f"result{num}.xlsx" for num in file_numbers]

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

# 青線時刻の辞書
blue_lines_dict = {num: [] for num in file_numbers}

# グラフの描画準備
fig, axes = plt.subplots(nrows=9, ncols=5, figsize=(25, 25), sharex=True, sharey=False)  # 行数を増やす
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 全体タイトルを追加
fig.suptitle('3sdata : F_lambda2 and Density/Traffic Volume for onramp = 750', fontsize=18, y=0.95)

legend_text = ""
plt.figtext(0.5, 0.01, legend_text, wrap=True, horizontalalignment='center', fontsize=12, color='black')

# 各ファイルとシートを処理
for col_idx, file_name in enumerate(file_names):
    file_path = os.path.join(base_path, file_name)
    file_number = int(file_name.replace('result', '').replace('.xlsx', ''))  # ファイル番号を取得
    blue_lines = blue_lines_dict[file_number]  # 青線時刻を取得

    # Sheet1 の Time, Density, traffic volume をプロット
    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_sheet1 = df_sheet1[['Time', 'Density', 'traffic volume']].dropna()
    ax = axes[0, col_idx]  # 各列の一番上のグラフ
    ax.plot(df_sheet1['Time'], df_sheet1['Density'], label='Density', color='green', linewidth=1.2)
    ax.plot(df_sheet1['Time'], df_sheet1['traffic volume'] * 10, label='Traffic Volume ×10', color='orange', linewidth=1.2)  # traffic volume を10倍
    ax.set_title(f'File {file_number} - Sheet1')
    ax.legend()
    ax.grid()


    for row_idx, (sheet, title) in enumerate(sheet_titles.items()):
        thin_red_lines, thick_red_lines = identify_red_lines(file_path, sheet)  # 赤線を引く時刻を決定
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df[['Time', 'F_lambda2']].dropna().sort_values('Time')
        df.columns = ['Time', 'Frambda2']

        ax = axes[row_idx + 1, col_idx]  # 行+1に設定
        ax.plot(df['Time'], df['Frambda2'], color='blue', linewidth=1)
        
        # 薄い赤線を引く
        #for thin_time in thin_red_lines:
            #ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)

        # 濃い赤線を引く
        for thick_time in thick_red_lines:
            ax.axvline(x=thick_time, color='red', linestyle='-', linewidth=1.5)

        # 青線を引く
        for blue_time in blue_lines:
            ax.axvline(x=blue_time, color='blue', linestyle='--', linewidth=1.2)

        ax.set_ylim(-120, 120)
        if col_idx == 0:
            ax.set_ylabel(f'Part {title}', fontsize=8)
        ax.grid()
        # Y軸の値0を目立たせる
        ax.axhline(y=0, color='black', linewidth=2)


plt.show()
