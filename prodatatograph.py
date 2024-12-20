import pandas as pd
import numpy as np
from scipy.linalg import eigh
import openpyxl
import matplotlib.pyplot as plt
import os

# === グローバル設定 ===
file_numbers = [5,6]  # 処理対象ファイル番号
base_path = "C://Users//YuheiTakada//OneDrive//デスクトップ//traffic-simulation-de"

# Laplacian行列の作成
def create_laplacian(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    L = np.diag(np.sum(A, axis=1)) - A
    return np.dot(
        np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))),
        np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))))
    )

# === 1. IGFT処理 ===
def process_igft(input_file, output_file):
    data = pd.read_excel(input_file, sheet_name="Sheet1", header=None)
    time_data = data.iloc[0, 0:].values
    speed_data = data.iloc[1:41, 0:].values
    row_100_data = data.iloc[99].values

    n = 40
    L_normalized = create_laplacian(n)
    eigenvalues, eigenvectors = eigh(L_normalized)

    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    reconstructed_data = []
    for t in range(speed_data.shape[1]):
        speed_at_t = speed_data[:, t]
        F_lambda = np.array([np.dot(speed_at_t, sorted_eigenvectors[:, i]) for i in range(n)])
        dominant_indices = np.argsort(np.abs(F_lambda))[-5:][::-1]
        reconstructed_signal = sum(F_lambda[i] * sorted_eigenvectors[:, i] for i in dominant_indices)
        reconstructed_data.append(reconstructed_signal)

    reconstructed_data = np.array(reconstructed_data).T
    output_df = pd.DataFrame(reconstructed_data, columns=time_data)
    output_df.loc[99] = row_100_data
    output_df.to_excel(output_file, index=False, header=True)

# === 2. reprodatatoresult処理 ===
def process_reprodatatoresult(input_file, result_file1, result_file2):
    workbook = openpyxl.load_workbook(input_file)
    sheet = workbook["Sheet1"]
    df1 = pd.read_excel(input_file, sheet_name="Sheet1", header=None)
    time_data = df1.iloc[0].values.flatten()
    speed_data = df1.iloc[41].values.flatten()

    n = 40
    L_normalized = create_laplacian(n)

    def calculate_F_lambda_real_parts(df):
        real_parts_list = []
        for column in df.columns:
            speed_data = df[column].values.flatten()
            eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)
            sorted_indices = np.argsort(eigenvalues)
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            F_lambda = np.zeros(sorted_eigenvectors.shape[1], dtype=complex)
            for i in range(sorted_eigenvectors.shape[1]):
                u_lambda_i = sorted_eigenvectors[:, i]
                F_lambda[i] = sum(speed_data * np.conj(u_lambda_i))

            real_parts = F_lambda.real
            real_parts_list.append(real_parts)
        return real_parts_list

    def write_sheet_with_ranks(writer, sheet_name, time_data, speed_data, real_parts_list):
        output_data = {
            'Time': list(time_data),
            'speed': list(speed_data)}
        for lambda_index in range(40):
            output_data[f'F_lambda{lambda_index+1}'] = [
                real_parts[lambda_index] if len(real_parts) > lambda_index else '' for real_parts in real_parts_list
            ]

        results_df = pd.DataFrame(output_data)
        results_df.to_excel(writer, index=False, sheet_name=sheet_name)

    with pd.ExcelWriter(result_file1, mode='w') as writer:
        real_parts_list = calculate_F_lambda_real_parts(df1)
        write_sheet_with_ranks(writer, "Sheet1", time_data, speed_data, real_parts_list)

# === 3. reresultgraph処理 ===
def plot_results(file_number, base_path):
    file_path = os.path.join(base_path, f"reresult{file_number}.xlsx")
    rank_file_path = os.path.join(base_path, f"reλrankresult{file_number}.xlsx")

    df_sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')[['Time', 'speed', 'F_lambda2']].dropna()
    df_sheet1.columns = ['Time', 'Speed', 'F_lambda2']

    thin_red_lines = df_sheet1.loc[df_sheet1['Speed'] < 50, 'Time'].tolist()

    def identify_yellow_lines(df):
        negative_indices = df['F_lambda2'] < 0
        groups = negative_indices.ne(negative_indices.shift()).cumsum()
        negative_streaks = df.groupby(groups).filter(lambda g: len(g) >= 5 and g['F_lambda2'].iloc[0] < 0)
        return negative_streaks['Time'].tolist()

    yellow_lines = identify_yellow_lines(df_sheet1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_sheet1['Time'], df_sheet1['Speed'], label='Speed', color='purple')
    for thin_time in thin_red_lines:
        ax.axvline(x=thin_time, color='red', linestyle='-', alpha=0.5)
    for yellow_time in yellow_lines:
        ax.axvline(x=yellow_time, color='yellow', linestyle='--', alpha=0.8)
    ax.set_title(f'File {file_number} Results')
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed / F_lambda2')
    ax.legend()
    plt.show()

# === 一連の処理実行 ===
for num in file_numbers:
    input_file = f'prodata{num}.xlsx'
    output_file = f'reprodata{num}.xlsx'
    result_file1 = f'reresult{num}.xlsx'
    result_file2 = f'reλrankresult{num}.xlsx'

    process_igft(input_file, output_file)
    process_reprodatatoresult(output_file, result_file1, result_file2)
    plot_results(num, base_path)
