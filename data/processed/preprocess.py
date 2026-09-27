import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(BASE_DIR, exist_ok=True)
raw_csv_path = os.path.join(BASE_DIR, '..', 'raw', 'AMZN[01.04.2021-12.31.2025].csv')
df = pd.read_csv(raw_csv_path)

df.columns = df.columns.str.strip()

if df['Price'].dtype == object:
    df['Price'] = df['Price'].str.replace(',', '').astype(float)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

processed_csv_path = os.path.join(BASE_DIR, 'AMZN_clean_prices.csv')
df.to_csv(processed_csv_path, index=False)

print(f"Cleaned data successfully created: {processed_csv_path}")
print(df[['Date', 'Price']].head())