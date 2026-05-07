import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df):
    """
    Calculates Recency, Frequency, and Monetary (Engagement) scores.
    Monetary is a proxy based on weighted events.
    """
    # 1. Define weights for engagement
    weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df['score'] = df['event'].map(weights)
    
    # 2. Get current timestamp (max date in dataset)
    current_date = df['timestamp'].max()
    
    # 3. Aggregate at visitorid level
    rfm = df.groupby('visitorid').agg({
        'timestamp': lambda x: (current_date - x.max()).days, # Recency
        'event': 'count',                                   # Frequency
        'score': 'sum'                                      # Monetary (Engagement)
    })
    
    # 4. Rename columns
    rfm.rename(columns={
        'timestamp': 'Recency',
        'event': 'Frequency',
        'score': 'Monetary'
    }, inplace=True)
    
    return rfm

def prepare_features(rfm_df):
    """
    Handles outliers and applies scaling.
    """
    # 1. Handle outliers/skewness with log transformation
    # Add small constant to avoid log(0) if any
    rfm_log = np.log1p(rfm_df)
    
    # 2. Apply StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)
    
    # 3. Return as DataFrame for convenience
    scaled_df = pd.DataFrame(
        scaled_features, 
        index=rfm_df.index, 
        columns=rfm_df.columns
    )
    
    return scaled_df
