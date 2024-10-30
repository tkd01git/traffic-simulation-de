import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. データの読み込み
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//010.xlsx"
sheet_name = '011'
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols='A:M')  # A列からM列までを読み込む

# 時刻データと対象データの取得
time_data = df.iloc[:, 0].values  # 1列目を時刻データと仮定
average_speed_data = df.iloc[:, 1].values  # 2列目を平均速度データと仮定

# 2. プロットの設定
plt.figure(figsize=(12, 8))

# 2列目のデータを黒線でプロット
plt.plot(time_data, df.iloc[:, 1].values, label='Column 2 (Black)', color='black', alpha=0.7)

# 3列目のデータを赤でプロット
plt.plot(time_data, df.iloc[:, 2].values, label='Column 3 (Red)', color='red', alpha=0.7)

# 4列目のデータを青でプロット
plt.plot(time_data, df.iloc[:, 3].values, label='Column 4 (Blue)', color='blue', alpha=0.7)

# 11列目のデータを黄色でプロット
plt.plot(time_data, df.iloc[:, 10].values, label='Column 11 (Yellow)', color='yellow', alpha=0.7)

# 12列目のデータを緑でプロット
plt.plot(time_data, df.iloc[:, 11].values, label='Column 12 (Green)', color='green', alpha=0.7)

# 13列目のデータを太線の紫でプロット
plt.plot(time_data, df.iloc[:, 12].values, label='Column 13 (Purple)', color='purple', linewidth=2.5)

# 縦軸の範囲を0から200に設定
plt.ylim(0, 200)

# グラフの設定
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Simulation with Colored and Styled Plots')
plt.legend()
plt.grid(True)

# プロットの表示
plt.show()
