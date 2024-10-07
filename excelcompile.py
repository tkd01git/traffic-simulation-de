import pandas as pd
import re
import sys
import os

def process_detector_data(file_path, speed_column_index):
    # エクセルファイルの読み込み
    df = pd.read_excel(file_path, engine='openpyxl')
    time_data = df.iloc[:, 0]
    data = df.iloc[:, speed_column_index]
    comments = df.iloc[:, 0]

    detector_data = {}
    detector_pattern = re.compile(r"#Detector (\d+)")
    current_detector = None
    current_time = []
    current_data = []

    for idx, comment in enumerate(comments):
        if isinstance(comment, str) and 'Detector' in comment:
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

    combined_data = pd.DataFrame()
    for detector, data in detector_data.items():
        if combined_data.empty:
            combined_data['Time'] = data['Time']
        combined_data[f'Detector_{detector}'] = data['Data']

    combined_data.replace("--", 0, inplace=True)
    combined_data = combined_data.transpose()
    combined_data = combined_data.iloc[:, 1:]
    return combined_data

def main():
    if len(sys.argv) < 3:
        print("Usage: python excelcompile.py <input_filename> <output_filename>")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    speed_combined_data = process_detector_data(input_file, 2)
    flow_combined_data = process_detector_data(input_file, 1)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        speed_combined_data.to_excel(writer, sheet_name="Sheet1", index=False, header=False)
        flow_combined_data.to_excel(writer, sheet_name="Sheet2", index=False, header=False)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
