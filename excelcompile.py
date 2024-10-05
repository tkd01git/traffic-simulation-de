import pandas as pd
import re

def process_detector_data(file_path, speed_column_index):
    # エクセルファイルの読み込み
    df = pd.read_excel(file_path, engine='openpyxl')

    # 必要なデータ（A列の時刻データ、指定した列のデータ）を抽出
    time_data = df.iloc[:, 0]  # A列 (0番目の列)
    data = df.iloc[:, speed_column_index]  # 指定された列
    comments = df.iloc[:, 0]  # コメントがA列にあると仮定（0番目の列）

    # detector番号ごとにデータを整理するための辞書
    detector_data = {}

    # コメントからdetector番号を抽出する正規表現
    detector_pattern = re.compile(r"#Detector (\d+)")

    current_detector = None
    current_time = []
    current_data = []

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
                        'Data': current_data
                    })

                # 新しいdetectorの番号に更新し、リストを初期化
                current_detector = detector_number
                current_time = []
                current_data = []
        
        # データ部分を処理（コメント行でない場合にデータを収集）
        elif pd.notna(time_data.iloc[idx]) and pd.notna(data.iloc[idx]):
            current_time.append(time_data.iloc[idx])
            current_data.append(data.iloc[idx])

    # 最後のdetectorのデータも保存
    if current_detector is not None and len(current_time) > 0:
        detector_data[current_detector] = pd.DataFrame({
            'Time': current_time,
            'Data': current_data
        })

    # すべてのdetectorのデータを一つのシートに統合
    combined_data = pd.DataFrame()

    # 各detectorごとのデータを時刻とデータごとにマージ
    for detector, data in detector_data.items():
        # 初回の処理：時刻データを最初の列に追加
        if combined_data.empty:
            combined_data['Time'] = data['Time']
        
        # detector番号ごとにデータを追加
        combined_data[f'Detector_{detector}'] = data['Data']

    # "--" を 0 に置換
    combined_data.replace("--", 0, inplace=True)

    # データを転置
    combined_data = combined_data.transpose()

    # 一列目を削除
    combined_data = combined_data.iloc[:, 1:]  # 一列目を削除

    return combined_data

# ファイルパス
file_path = "C://Users//YuheiTakada//Downloads//1004simulation.xlsx"

# 速度データの処理
speed_combined_data = process_detector_data(file_path, 4)

# 流量データの処理
flow_combined_data = process_detector_data(file_path, 2)

# エクセルにデータを書き込む
with pd.ExcelWriter("sorted_data.xlsx", engine='openpyxl') as writer:
    # 速度データをSheet1に書き込み
    speed_combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False, startrow=0, startcol=0)

    # 流量データをSheet2に書き込み
    flow_combined_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False, startrow=0, startcol=0)

# 読み込み直し（保存されたデータに対して処理を行うため）
with pd.ExcelFile("sorted_data.xlsx", engine='openpyxl') as reader:
    # Sheet1（速度データ）とSheet2（流量データ）を読み込む
    sheet1 = pd.read_excel(reader, sheet_name="Sheet1", header=None)
    sheet2 = pd.read_excel(reader, sheet_name="Sheet2", header=None)

    # 必要な行数を確認し、22行目と23行目を追加
    required_rows = 24  # 最低でも24行必要
    current_rows = sheet2.shape[0]

    # 行が足りない場合は、必要な行数まで行を追加する
    if current_rows < required_rows:
        additional_rows = required_rows - current_rows
        new_rows = pd.DataFrame(0, index=range(additional_rows), columns=sheet2.columns)
        sheet2 = pd.concat([sheet2, new_rows], ignore_index=True)

    # 各列の2〜21行目の値を掛け合わせてその列の総和を求める
    for col in range(sheet1.shape[1]):  # 列の数だけループ
        # 2〜21行目の掛け合わせ
        product_sum = (sheet1.iloc[1:21, col] * sheet2.iloc[1:21, col]).sum()
        
        # 結果をSheet2の22行目に記入
        sheet2.iloc[21, col] = product_sum

    # 各列の2〜21行目の総和を計算して23行目に記入
    for col in range(sheet2.shape[1]):
        # 2〜21行目の総和を計算
        col_sum = sheet2.iloc[1:21, col].sum()
        
        # 結果をSheet2の23行目に記入
        sheet2.iloc[22, col] = col_sum

        # 22行目の値を23行目の値で割った結果を24行目に記入（小数第一位まで表示）
        if col_sum != 0:  # ゼロ除算を避けるため
            sheet2.iloc[23, col] = round(sheet2.iloc[21, col] / col_sum, 1)  # 小数第一位までの表示
        else:
            sheet2.iloc[23, col] = 0  # ゼロ除算の場合は0を設定

    # 最後の列を削除
    sheet1 = sheet1.drop(sheet1.columns[-1], axis=1)
    sheet2 = sheet2.drop(sheet2.columns[-1], axis=1)

# 更新されたデータを再度エクセルファイルに保存
with pd.ExcelWriter("sorted_data.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    sheet2.to_excel(writer, sheet_name="Sheet2", index=False, header=False, startrow=0, startcol=0)

# 列名をアルファベットに変換するための関数
def convert_to_excel_column(index):
    """インデックスをエクセルの列名（A, B, C, ...）に変換する関数"""
    column_name = ""
    while index >= 0:
        column_name = chr(index % 26 + ord('A')) + column_name
        index = index // 26 - 1
    return column_name

# 列名とその最初の行の値を対応させて表示
print("列名と最初の行の値:")
for col in range(sheet2.shape[1]):
    col_name = convert_to_excel_column(col)  # 列名を取得
    first_row_value = sheet2.iloc[0, col]  # 1行目の値を取得
    print(f"{col_name}: {first_row_value}")  # 対応を表示

# 保存したファイル名を表示
print("保存したファイル名: sorted_data.xlsx")
