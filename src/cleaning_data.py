import pandas as pd
import numpy as np

def clean_data(df):
    print(f"Original shape: {df.shape}")
    
    df = df.drop_duplicates()
    df = df.dropna()
    
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], unit='ms')
        
    print(f"Cleaned shape: {df.shape}")
    print("="*30)
    
    return df