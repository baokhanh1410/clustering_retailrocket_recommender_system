"""
features.py — RFM Feature Engineering & Preprocessing Module

This module provides functions for creating customer-level RFM (Recency,
Frequency, Monetary) features from raw event data, and for preprocessing
those features (outlier handling via log transformation and standardization)
to prepare them for clustering algorithms.

Key Exports:
    - calculate_rfm(df): Computes RFM metrics per visitor using DuckDB SQL.
    - prepare_features(rfm_df): Applies log1p transformation + StandardScaler
      to numeric RFM columns, making them suitable for distance-based models.
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Constants
# =============================================================================

# Columns that represent the actual RFM features to be transformed.
# 'visitorid' and other non-feature columns are excluded from scaling.
RFM_FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary']

# =============================================================================
# Public Functions
# =============================================================================


def calculate_rfm(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each visitor.

    Uses DuckDB SQL to efficiently aggregate the cleaned event DataFrame
    into one row per visitor with three behavioral metrics:
      - Recency:   Number of days since the visitor's last interaction
                   (relative to the max timestamp in the dataset).
      - Frequency:  Total count of interactions (view / addtocart / transaction).
      - Monetary:   Weighted engagement score where each event type contributes
                   a different weight (view=1, addtocart=3, transaction=5).
                   This proxy replaces actual monetary value since price data
                   is unavailable in the Retailrocket dataset.

    Args:
        df (pd.DataFrame): Cleaned event DataFrame containing at least
            'visitorid', 'timestamp', and 'event' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns ['visitorid', 'Recency',
            'Frequency', 'Monetary'], one row per unique visitor.
    """

    # ------------------------------------------------------------------
    # The CTE `max_time` calculates the latest timestamp in the entire
    # dataset, which serves as the "current date" reference point for
    # computing Recency.
    #
    # Recency = days between a visitor's LAST interaction and max_time.
    #   → A high Recency value means the visitor has been INACTIVE for
    #     a long time (potential churn risk).
    #   → A low Recency value means the visitor was recently active.
    #
    # Frequency = simple COUNT of all events for the visitor.
    #
    # Monetary = weighted sum using CASE expression to assign:
    #   'view'        → 1 point   (low engagement)
    #   'addtocart'   → 3 points  (medium engagement)
    #   'transaction'  → 5 points  (high engagement)
    # ------------------------------------------------------------------
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
    Preprocess RFM features for clustering: log-transform then standardize.

    RFM distributions (especially Frequency and Monetary) are typically
    heavily right-skewed.  This two-step pipeline addresses that:
      1. Log transformation (log1p) reduces skewness and compresses
         extreme outliers, making the distribution more symmetric.
      2. StandardScaler (z-score normalization) ensures all three RFM
         dimensions contribute equally to distance-based clustering
         algorithms like K-Means.

    Only the numeric feature columns (Recency, Frequency, Monetary) are
    transformed. Non-feature columns such as 'visitorid' are automatically
    excluded to prevent nonsensical transformations.

    Args:
        rfm_df (pd.DataFrame): DataFrame containing at least the columns
            'Recency', 'Frequency', and 'Monetary'. May also contain
            other columns (e.g., 'visitorid') which will be ignored.

    Returns:
        pd.DataFrame: A DataFrame with only the RFM feature columns,
            log-transformed and standardized, indexed the same as the input.
    """

    # ------------------------------------------------------------------
    # Step 1: Select only the numeric RFM feature columns.
    #   This prevents accidentally transforming identifier columns
    #   like 'visitorid', which would produce meaningless values.
    # ------------------------------------------------------------------
    feature_cols = [col for col in RFM_FEATURE_COLUMNS if col in rfm_df.columns]
    rfm_subset = rfm_df[feature_cols]

    # ------------------------------------------------------------------
    # Step 2: Apply log1p (log(1 + x)) transformation.
    #   log1p is preferred over log because it handles zero values safely
    #   (log(0) is undefined, but log1p(0) = 0).
    # ------------------------------------------------------------------
    rfm_log = np.log1p(rfm_subset)

    # ------------------------------------------------------------------
    # Step 3: Standardize with StandardScaler (mean=0, std=1).
    #   This ensures that no single RFM dimension dominates the
    #   Euclidean distance computation used by K-Means.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    # ------------------------------------------------------------------
    # Step 4: Return as a DataFrame with the original index preserved.
    # ------------------------------------------------------------------
    scaled_df = pd.DataFrame(
        scaled_features,
        index=rfm_df.index,
        columns=feature_cols
    )

    return scaled_df
