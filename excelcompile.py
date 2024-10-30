import pandas as pd
import openpyxl
import re
import os

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

# ファイルパス
file_path = "C://Users//YuheiTakada//Downloads//data (4).xlsx"

# 速度データの処理
speed_combined_data = process_detector_data(file_path, 2)

# 流量データの処理
flow_combined_data = process_detector_data(file_path, 1)

# エクセルにデータを書き込む
with pd.ExcelWriter("prodata.xlsx", engine='openpyxl') as writer:
    if not speed_combined_data.empty:
        speed_combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    if not flow_combined_data.empty:
        flow_combined_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)

# Sheet2のデータを処理
sheet2_data = pd.read_excel("prodata.xlsx", sheet_name="Sheet2", header=None, engine='openpyxl')

# Sheet1のデータを再度読み込む
sheet1_data = pd.read_excel("prodata.xlsx", sheet_name="Sheet1", header=None, engine='openpyxl')

# 必要な行数を確認し、不足している場合は行を追加
required_rows = 50
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
    
# 各列の2~11行目で、sheet1_dataとsheet2_dataのセルを掛け合わせて、その総和を25行目に書き込む
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[1:11, col] * sheet2_data.iloc[1:11, col]).sum()
    sheet2_data.iloc[24, col] = product_sum

# 各列のsheet2_dataの2~11行目の総和を26行目に書き込む
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[1:11, col].sum()
    sheet2_data.iloc[25, col] = col_sum

# 各列の25行目を26行目で割り、小数第2位で四捨五入して27行目に書き込む
for col in range(sheet2_data.shape[1]):
    value_25 = sheet2_data.iloc[24, col]
    value_26 = sheet2_data.iloc[25, col]
    
    if pd.notna(value_25) and pd.notna(value_26) and value_26 != 0:
        result = round(value_25 / value_26, 2)
    else:
        result = None
    
    sheet2_data.iloc[26, col] = result
    
# 各列の12~21行目で、sheet1_dataとsheet2_dataのセルを掛け合わせて、その総和を28行目に書き込む
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[11:21, col] * sheet2_data.iloc[11:21, col]).sum()
    sheet2_data.iloc[27, col] = product_sum

# 各列のsheet2_dataの12~21行目の総和を29行目に書き込む
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[11:21, col].sum()
    sheet2_data.iloc[28, col] = col_sum

# 各列の28行目を29行目で割り、小数第2位で四捨五入して30行目に書き込む
for col in range(sheet2_data.shape[1]):
    value_28 = sheet2_data.iloc[27, col]
    value_29 = sheet2_data.iloc[28, col]
    
    if pd.notna(value_28) and pd.notna(value_29) and value_29 != 0:
        result = round(value_28 / value_29, 2)
    else:
        result = None
    
    sheet2_data.iloc[29, col] = result
    
# 各列の2~6行目で、sheet1_dataとsheet2_dataのセルを掛け合わせて、その総和を31行目に書き込む
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[1:6, col] * sheet2_data.iloc[1:6, col]).sum()
    sheet2_data.iloc[30, col] = product_sum

# 各列のsheet2_dataの2~6行目の総和を32行目に書き込む
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[1:6, col].sum()
    sheet2_data.iloc[31, col] = col_sum

# 各列の31行目を32行目で割り、小数第2位で四捨五入して33行目に書き込む
for col in range(sheet2_data.shape[1]):
    value_31 = sheet2_data.iloc[30, col]
    value_32 = sheet2_data.iloc[31, col]
    
    if pd.notna(value_31) and pd.notna(value_32) and value_32 != 0:
        result = round(value_31 / value_32, 2)
    else:
        result = None
    
    sheet2_data.iloc[32, col] = result
    

# 各列の7~16行目で、sheet1_dataとsheet2_dataのセルを掛け合わせて、その総和を34行目に書き込む
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[6:16, col] * sheet2_data.iloc[6:16, col]).sum()
    sheet2_data.iloc[33, col] = product_sum

# 各列のsheet2_dataの7~16行目の総和を35行目に書き込む
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[6:16, col].sum()
    sheet2_data.iloc[34, col] = col_sum

# 各列の34行目を35行目で割り、小数第2位で四捨五入して36行目に書き込む
for col in range(sheet2_data.shape[1]):
    value_34 = sheet2_data.iloc[33, col]
    value_35 = sheet2_data.iloc[34, col]
    
    if pd.notna(value_34) and pd.notna(value_35) and value_35 != 0:
        result = round(value_34 / value_35, 2)
    else:
        result = None
    
    sheet2_data.iloc[35, col] = result

# 各列の17~21行目で、sheet1_dataとsheet2_dataのセルを掛け合わせて、その総和を37行目に書き込む
for col in range(sheet1_data.shape[1]):
    product_sum = (sheet1_data.iloc[16:21, col] * sheet2_data.iloc[16:21, col]).sum()
    sheet2_data.iloc[36, col] = product_sum

# 各列のsheet2_dataの17~21行目の総和を38行目に書き込む
for col in range(sheet2_data.shape[1]):
    col_sum = sheet2_data.iloc[16:21, col].sum()
    sheet2_data.iloc[37, col] = col_sum

# 各列の37行目を38行目で割り、小数第2位で四捨五入して39行目に書き込む
for col in range(sheet2_data.shape[1]):
    value_37 = sheet2_data.iloc[36, col]
    value_38 = sheet2_data.iloc[37, col]
    
    if pd.notna(value_37) and pd.notna(value_38) and value_38 != 0:
        result = round(value_37 / value_38, 2)
    else:
        result = None
    
    sheet2_data.iloc[38, col] = result



    
    
# 最後に全てのデータを書き込む
with pd.ExcelWriter("prodata.xlsx", engine='openpyxl', mode='w') as writer:
    sheet1_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
    sheet2_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)

print("Data has been successfully written to 'prodata.xlsx'.")

