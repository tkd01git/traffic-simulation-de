import pandas as pd
import openpyxl
import re
import os


# 使用するデータファイル
file_path = "C://Users//Yuheitakada//Downloads//data (71).xlsx"

# データ記録エクセルファイル
EXCEL_FILE_NAME = "prodata58.xlsx"  # 任意の名前に変更可能


def process_detector_data(file_path, speed_column_index):
    # Excelファイルからすべてのデータを読み込む
    df = pd.read_excel(file_path, sheet_name="sheet1", engine='openpyxl')
    
    # 指定した列がデータの範囲を超える場合は空のDataFrameを返す
    if speed_column_index >= df.shape[1]:
        return pd.DataFrame()

    # 各カラムからデータを取得
    time_data = df.iloc[:, 0]
    data = df.iloc[:, speed_column_index]
    comments = df.iloc[:, 0]

    detector_data = {}
    detector_pattern = re.compile(r"detector(\d+)", re.IGNORECASE)

    detector_data['0'] = pd.DataFrame({
        'Time': time_data.iloc[:].values,  # 全行を取得
        'Data': data.iloc[:].values
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


# 速度データの処理
speed_combined_data = process_detector_data(file_path, 2)

# 流量データの処理
flow_combined_data = process_detector_data(file_path, 1)

# エクセルにデータを書き込む
with pd.ExcelWriter(EXCEL_FILE_NAME, engine='openpyxl') as writer:
    if not speed_combined_data.empty:
        speed_combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    if not flow_combined_data.empty:
        flow_combined_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)

# Sheet2のデータを処理
sheet2_data = pd.read_excel(EXCEL_FILE_NAME, sheet_name="Sheet2", header=None, engine='openpyxl')

# Sheet1のデータを再度読み込む
sheet1_data = pd.read_excel(EXCEL_FILE_NAME, sheet_name="Sheet1", header=None, engine='openpyxl')

# 必要な行数を確認し、不足している場合は行を追加
required_rows = 100
if sheet2_data.shape[0] < required_rows:
    additional_rows = required_rows - sheet2_data.shape[0]
    sheet2_data = pd.concat([sheet2_data, pd.DataFrame([[None] * sheet2_data.shape[1]] * additional_rows, columns=sheet2_data.columns)], ignore_index=True)
    

# 追加：4列目のデータをSheet2の100行目に書き込む
df_additional_data = pd.read_excel(file_path, sheet_name="sheet1", engine='openpyxl')
if df_additional_data.shape[1] >= 4:
    sheet2_data.iloc[99, :] = df_additional_data.iloc[:, 3].values[:sheet2_data.shape[1]]

# 最後に全てのデータを書き込む
with pd.ExcelWriter(EXCEL_FILE_NAME, engine='openpyxl', mode='w') as writer:
    sheet1_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    sheet2_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)

print(f"Data has been successfully written to '{EXCEL_FILE_NAME}'.")
