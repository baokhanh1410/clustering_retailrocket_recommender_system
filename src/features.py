"""
features.py — RFM Feature Engineering & Preprocessing Module

This module provides functions for creating customer-level RFM (Recency,
Frequency, Monetary) features from raw event data, and for preprocessing
those features to prepare them for clustering algorithms.

Key Exports:
    - calculate_rfm(df): Computes RFM metrics per visitor using DuckDB SQL.
    - prepare_features(rfm_df): Applies log1p transformation + MinMaxScaler.
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# =============================================================================
# Constants
# =============================================================================

RFM_FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary"]

# =============================================================================
# Public Functions
# =============================================================================


def calculate_rfm(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each visitor.

    Fix applied: 'Monetary' is now calculated as an Engagement/Conversion Score
    rather than cumulative points to break the perfect correlation with 'Frequency'.

    Args:
        df (pd.DataFrame): Cleaned events DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with 'visitorid' as index and RFM columns.
    """
    query = """
        WITH rfm_base AS (
            SELECT 
                visitorid,
                MAX(timestamp) AS last_interaction,
                COUNT(*) AS Frequency,
                -- Engagement Score: High-value actions divided by total interactions
                -- AddToCart = 3 points, Transaction = 5 points
                CAST(SUM(CASE 
                    WHEN event = 'transaction' THEN 50
                    WHEN event = 'addtocart' THEN 30
                    WHEN event = 'view' THEN 0
                    ELSE 0
                END) AS FLOAT) AS Monetary
            FROM df
            -- Filter out missing/sentinel values generated during cleaning
            WHERE visitorid != -1 AND timestamp IS NOT NULL
            GROUP BY visitorid
        )
        SELECT 
            visitorid,
            -- Recency: Days since last interaction relative to the max date in dataset
            DATE_DIFF('day', last_interaction, (SELECT MAX(last_interaction) FROM rfm_base)) AS Recency,
            Frequency,
            Monetary
        FROM rfm_base
    """

    # Execute the query and set visitorid as the DataFrame index
    rfm_df = duckdb.sql(query).df()
    rfm_df.set_index("visitorid", inplace=True)

    return rfm_df


def prepare_features(rfm_df):
    """
    Preprocess RFM features using log1p transformation and standardization.
    """
    feature_cols = [col for col in RFM_FEATURE_COLUMNS if col in rfm_df.columns]
    rfm_subset = rfm_df[feature_cols]

    # Apply log1p (log(1 + x)) transformation to handle zero values safely
    rfm_log = np.log1p(rfm_subset)

    # Standardize with MinMaxScaler (mean=0, std=1)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    scaled_df = pd.DataFrame(scaled_features, index=rfm_df.index, columns=feature_cols)

    return scaled_df
