import pandas as pd
import re

# エクセルファイルの読み込み
file_path = "C://Users//YuheiTakada//OneDrive//ドキュメント//0926simulation.xlsx"  # ファイルのパスを指定してください
df = pd.read_excel(file_path, engine='openpyxl')

# 必要なデータ（A列の時刻データ、E列の速度データ）を抽出
time_data = df.iloc[:, 0]  # A列 (0番目の列)
speed_data = df.iloc[:, 4]  # E列 (4番目の列)
comments = df.iloc[:, 0]  # コメントがA列にあると仮定（0番目の列）

# detector番号ごとにデータを整理するための辞書
detector_data = {}

# コメントからdetector番号を抽出する正規表現
detector_pattern = re.compile(r"#Detector (\d+)")

current_detector = None
current_time = []
current_speed = []

for idx, comment in enumerate(comments):
    # コメントが存在する場合（コメント行であるかどうかを判定）
    if isinstance(comment, str) and 'Detector' in comment:
        # detector番号が記載されているコメントを検出
        match = detector_pattern.search(comment)
        if match:
            detector_number = match.group(1)

            # 前のdetectorのデータを保存
            if current_detector is not None and len(current_time) > 0:
                detector_data[current_detector] = pd.DataFrame({
                    'Time': current_time,
                    'Speed': current_speed
                })

            # 新しいdetectorの番号に更新し、リストを初期化
            current_detector = detector_number
            current_time = []
            current_speed = []
    
    # データ部分を処理（コメント行でない場合にデータを収集）
    elif pd.notna(time_data.iloc[idx]) and pd.notna(speed_data.iloc[idx]):
        current_time.append(time_data.iloc[idx])
        current_speed.append(speed_data.iloc[idx])

# 最後のdetectorのデータも保存
if current_detector is not None and len(current_time) > 0:
    detector_data[current_detector] = pd.DataFrame({
        'Time': current_time,
        'Speed': current_speed
    })

# すべてのdetectorのデータを一つのシートに統合
combined_data = pd.DataFrame()

# 各detectorごとのデータを時刻と速度ごとにマージ
for detector, data in detector_data.items():
    # 初回の処理：時刻データを最初の列に追加
    if combined_data.empty:
        combined_data['Time'] = data['Time']
    
    # detector番号ごとに速度データを追加
    combined_data[f'Detector_{detector}'] = data['Speed']

# "--" を 0 に置換
combined_data.replace("--", 0, inplace=True)

# データを転置
combined_data = combined_data.transpose()

# エクセルにデータを書き込む (Combined_Detectors シートのみ)
with pd.ExcelWriter("sorted_speed_data.xlsx", engine='openpyxl') as writer:
    # すべてのdetectorのデータを一つのシートにまとめて書き込む（ヘッダーを削除）
    combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)

print("データのソートが完了し、sorted_speed_data.xlsxに保存されました。")
