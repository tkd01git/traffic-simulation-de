import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. データの読み込み
file_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//001.xlsx"
sheet_name = '001'
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols='A:Z')  # 適切な列範囲を指定

# 時刻データと対象データの取得
time_data = df.iloc[:, 0].values  # 1列目を時刻データと仮定
average_speed_data = df.iloc[:, 1].values  # 2列目を平均速度データと仮定

# 処理を行うデータ（4, 5, 6列目）
signal_columns = [np.abs(df.iloc[:, 21]), np.abs(df.iloc[:, 20]), np.abs(df.iloc[:, 19])]  # 4, 5, 6列目のデータを取得

# 2. フィルタリングとプロットの設定
plt.figure(figsize=(12, 8))

# 2列目のデータを黒線でプロット
plt.plot(time_data, average_speed_data, label='Average Speed', color='black', alpha=0.7)

# 4〜6列目の各信号データに対してフィルタリングを適用
for idx, signal_data in enumerate(signal_columns):
    # 離散フーリエ変換 (DFT)
    fft_values = np.fft.fft(signal_data.values)
    frequencies = np.fft.fftfreq(len(signal_data), d=(time_data[1] - time_data[0]))

    # ノイズ除去のためのフィルタリング
    # 高周波成分をゼロにする
    cutoff_freq = 0.1  # カットオフ周波数の設定 (例: 0.1 Hz)
    fft_filtered = np.where(np.abs(frequencies) > cutoff_freq, 0, fft_values)

    # 逆DFTで信号に戻す
    filtered_signal = np.fft.ifft(fft_filtered)

    # フィルタリング後の信号データをプロット
    plt.plot(time_data, filtered_signal.real, label=f'Filtered Signal (Column {idx + 4})', alpha=0.7)

# 縦軸の範囲を0から200に設定
plt.ylim(0, 200)

# 6. 平均速度が40を下回っている時刻の領域を赤で塗りつぶす
plt.fill_between(time_data, 0, 200, where=(average_speed_data < 40), color='red', alpha=0.1, label='Below Y = 40')

# グラフの設定
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Simulation with Multiple Signal Filtering')
plt.legend()
plt.grid(True)

# プロットの表示
plt.show()
