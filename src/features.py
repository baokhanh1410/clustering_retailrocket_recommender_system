import pandas as pd
import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df):
    """
    Calculates RFM.
    """
    query = """
    WITH max_time AS (
        SELECT MAX(timestamp) AS current_date FROM df
    )
    SELECT 
        visitorid,
        -- Recency: Diff days between latest interaction and current date
        date_diff('day', MAX(timestamp), (SELECT current_date FROM max_time)) AS Recency,
        
        -- Frequency: Count total interactions
        COUNT(event) AS Frequency,
        
        -- Monetary: Get sum of weights
        SUM(
            CASE event
                WHEN 'view' THEN 1
                WHEN 'addtocart' THEN 3
                WHEN 'transaction' THEN 5
                ELSE 0
            END
        ) AS Monetary
    FROM df
    GROUP BY visitorid
    """

    return duckdb.sql(query).df()


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
