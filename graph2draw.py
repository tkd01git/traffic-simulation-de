import pandas as pd
import matplotlib.pyplot as plt
import os

# 設定
file_numbers = [11, 12, 13, 14, 15]
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"
result_title = "Result Data Visualization"
rankresult_title = "λrankresult Data Visualization"
part_labels = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5', 'Part 6', 'Part 7', 'Part 8']

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

# 1つ目の処理（resultx.xlsx）
fig1, axes1 = plt.subplots(nrows=9, ncols=5, figsize=(25, 20), sharex=True, sharey=False)
fig1.subplots_adjust(hspace=0.5, wspace=0.3)
fig1.suptitle(result_title, fontsize=16, fontweight='bold')  # 大きなタイトル

for col_idx, file_number in enumerate(file_numbers):
    file_path = os.path.join(base_path, f"result{file_number}.xlsx")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Sheet1 の速度をプロット
    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_sheet1 = df_sheet1[['Time', 'speed']].dropna()
    thin_red_lines = identify_red_lines(file_path, 'Sheet1', speed_threshold=50)

    ax = axes1[0, col_idx]
    ax.plot(df_sheet1['Time'], df_sheet1['speed'], label='Speed', color='purple', linewidth=1.2)
    for thin_time in thin_red_lines:
        ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
    ax.set_title(f"File {file_number} - Speed")
    ax.legend()
    ax.grid()

    # Sheet2~Sheet8 の F_lambda2 をプロット
    for row_idx in range(1, 9):  # 行番号
        df = pd.read_excel(file_path, sheet_name=f'Sheet{row_idx + 1}')
        df = df[['Time', 'F_lambda2']].dropna().sort_values('Time')
        df.columns = ['Time', 'Frambda2']

        ax = axes1[row_idx, col_idx]
        ax.plot(df['Time'], df['Frambda2'], color='blue', linewidth=1)
        for thin_time in thin_red_lines:
            ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
        if col_idx == 0:  # 左端の列にのみ Part ラベルを追加
            ax.set_ylabel(part_labels[row_idx - 1], fontsize=10, fontweight='bold')
        ax.grid()
        ax.axhline(y=0, color='black', linewidth=1.2)

plt.show()

# 2つ目の処理（λrankresultx.xlsx）
fig2, axes2 = plt.subplots(nrows=9, ncols=5, figsize=(25, 20), sharex=True, sharey=False)
fig2.subplots_adjust(hspace=0.5, wspace=0.3)
fig2.suptitle(rankresult_title, fontsize=16, fontweight='bold')  # 大きなタイトル

for col_idx, file_number in enumerate(file_numbers):
    file_path = os.path.join(base_path, f"λrankresult{file_number}.xlsx")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Sheet1 の速度をプロット
    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
    df_sheet1 = df_sheet1[['Time', 'speed']].dropna()
    thin_red_lines = identify_red_lines(file_path, 'Sheet1', speed_threshold=50)

    ax = axes2[0, col_idx]
    ax.plot(df_sheet1['Time'], df_sheet1['speed'], label='Speed', color='purple', linewidth=1.2)
    for thin_time in thin_red_lines:
        ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
    ax.set_title(f"File {file_number} - Speed")
    ax.legend()
    ax.grid()

    # Sheet2~Sheet8 の Rank of F_lambda2 をプロット
    for row_idx in range(1, 9):  # 行番号
        df = pd.read_excel(file_path, sheet_name=f'Sheet{row_idx + 1}')
        if 'Rank of F_lambda2' not in df.columns:
            print(f"'Rank of F_lambda2' not found in sheet {row_idx + 1} of {file_path}")
            continue
        df = df[['Time', 'Rank of F_lambda2']].dropna().sort_values('Time')
        df.columns = ['Time', 'Rank of Frambda2']

        ax = axes2[row_idx, col_idx]
        ax.plot(df['Time'], df['Rank of Frambda2'], color='blue', linewidth=1)
        for thin_time in thin_red_lines:
            ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5, linewidth=1.2)
        if col_idx == 0:  # 左端の列にのみ Part ラベルを追加
            ax.set_ylabel(part_labels[row_idx - 1], fontsize=10, fontweight='bold')
        ax.grid()
        ax.axhline(y=0, color='black', linewidth=1.2)

plt.show()
