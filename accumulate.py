import pandas as pd
import os

# 元のExcelファイルのパスを格納
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//λrank_"
file_numbers = [f"{i:03d}" for i in range(0, 1100, 100)]  # 000, 100, ..., 900
input_files = [f"{base_path}{num}.xlsx" for num in file_numbers]
output_file = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de//100plusF3.xlsx"

# ExcelWriterで統合
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for num, file in zip(file_numbers, input_files):
        if os.path.exists(file):  # ファイルが存在するか確認
            # 明示的にエンジンを指定してファイルを読み込む
            try:
                df = pd.read_excel(file, engine='openpyxl')
                df.to_excel(writer, sheet_name=num, index=False)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        else:
            print(f"File not found: {file}")

print(f"Files successfully merged into {output_file}")
