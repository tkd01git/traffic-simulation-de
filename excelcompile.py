import pandas as pd
import re

def process_detector_data(file_path, speed_column_index):
    df = pd.read_excel(file_path, sheet_name="sheet1", engine='openpyxl')
    
    if speed_column_index >= df.shape[1]:
        return pd.DataFrame()
        
    time_data = df.iloc[:, 0]
    data = df.iloc[:, speed_column_index]
    comments = df.iloc[:, 0]

    detector_data = {}
    detector_pattern = re.compile(r"detector(\d+)", re.IGNORECASE)

    detector_data['0'] = pd.DataFrame({
        'Time': time_data.iloc[0:250].values,
        'Data': data.iloc[0:250].values
    })

    current_detector = None
    current_time = []
    current_data = []

    for idx, comment in enumerate(comments):
        if isinstance(comment, str) and re.search(detector_pattern, comment):
            match = detector_pattern.search(comment)
            if match:
                detector_number = match.group(1)

                if current_detector is not None and len(current_time) > 0:
                    detector_data[current_detector] = pd.DataFrame({
                        'Time': current_time,
                        'Data': current_data
                    })

                current_detector = detector_number
                current_time = []
                current_data = []
        
        elif pd.notna(time_data.iloc[idx]) and pd.notna(data.iloc[idx]):
            current_time.append(time_data.iloc[idx])
            current_data.append(data.iloc[idx])

    if current_detector is not None and len(current_time) > 0:
        detector_data[current_detector] = pd.DataFrame({
            'Time': current_time,
            'Data': current_data
        })

    if not detector_data:
        return pd.DataFrame()

    combined_data = pd.DataFrame()

    for detector, data in detector_data.items():
        if combined_data.empty:
            combined_data['Time'] = data['Time']
        
        combined_data[f'Detector_{detector}'] = data['Data']
    
    combined_data.replace("--", 0, inplace=True)
    combined_data = combined_data.transpose()
    combined_data.reset_index(drop=True, inplace=True)

    return combined_data

def column_to_letter(column_index):
    """Excelの列番号をアルファベットに変換"""
    letter = ""
    while column_index >= 0:
        letter = chr(column_index % 26 + ord('A')) + letter
        column_index = column_index // 26 - 1
    return letter

# ファイルパス
file_path = "C://Users//YuheiTakada//Downloads//data.xlsx"

# 速度データの処理
speed_combined_data = process_detector_data(file_path, 2)

# 流量データの処理
flow_combined_data = process_detector_data(file_path, 1)

# エクセルにデータを書き込む
with pd.ExcelWriter("prodata.xlsx", engine='openpyxl') as writer:
    if not speed_combined_data.empty:
        speed_combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False, startrow=0, startcol=0)
    if not flow_combined_data.empty:
        flow_combined_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False, startrow=0, startcol=0)

# Sheet2のデータを処理
sheet2_data = pd.read_excel("prodata.xlsx", sheet_name="Sheet2", header=None)

# Sheet1のデータを再度読み込む
sheet1_data = pd.read_excel("prodata.xlsx", sheet_name="Sheet1", header=None)

# 必要な行数を確認し、不足している場合は行を追加
required_rows = 24
if sheet2_data.shape[0] < required_rows:
    additional_rows = required_rows - sheet2_data.shape[0]
    sheet2_data = pd.concat([sheet2_data, pd.DataFrame([[None] * sheet2_data.shape[1]] * additional_rows, columns=sheet2_data.columns)], ignore_index=True)

# 各列の2〜21行目の値を掛け合わせてその列の総和を求める
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[1:21, col] * sheet2_data.iloc[1:21, col]).sum()
    sheet2_data.iloc[21, col] = product_sum

# 各列の2〜21行目の総和を計算して23行目に記入
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[1:21, col].sum()
    sheet2_data.iloc[22, col] = col_sum

# 各列の22行目を23行目で割り、小数第2位で四捨五入して24行目に記入
for col in range(sheet2_data.shape[1]):
    value_22 = sheet2_data.iloc[21, col]
    value_23 = sheet2_data.iloc[22, col]
    
    if pd.notna(value_22) and pd.notna(value_23) and value_23 != 0:
        result = round(value_22 / value_23, 2)
    else:
        result = None
    
    sheet2_data.iloc[23, col] = result

# 初めて平均速度が40km/hを下回った時をチェック
for col in range(sheet2_data.shape[1]):
    avg_speed = sheet2_data.iloc[23, col]  # 24行目のデータが平均速度
    if pd.notna(avg_speed) and avg_speed < 40:
        current_column_letter = column_to_letter(col)
        first_row_value = sheet2_data.iloc[0, col]  # その列の第一行を取得

        # 10列前と60列前の列名を取得
        previous_10_column_letter = column_to_letter(col - 10) if col >= 10 else "N/A"
        previous_60_column_letter = column_to_letter(col - 60) if col >= 60 else "N/A"

        # 結果の出力
        print(f"jam occuring time: {first_row_value}")
        print(f"60 columns before: {previous_60_column_letter}")
        print(f"10 columns before: {previous_10_column_letter}")
        break  # 最初に40km/hを下回った列を見つけたら終了


# 最終的なデータをExcelに保存
with pd.ExcelWriter("prodata.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    sheet1_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    sheet2_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)
