import pandas as pd
import os
import zipfile

def extract_and_load_data(zip_path='dt_health.zip', extract_path='data/'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    all_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.csv') or file.endswith('.xlsx'):
                all_files.append(os.path.join(root, file))
    full_df = pd.DataFrame()
    for file_path in all_files:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        full_df = pd.concat([full_df, df], ignore_index=True)
    full_df.dropna(inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df

if __name__ == "__main__":
    df = extract_and_load_data() 