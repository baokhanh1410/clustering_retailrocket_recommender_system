"""
cleaning_data.py — Data Cleaning Module via DuckDB SQL

This module provides a function to clean a raw pandas DataFrame using DuckDB's
in-process SQL engine.  The cleaning pipeline handles:
  - Missing value imputation (NaN → 0 for transactionid, -1 for others)
  - Timestamp conversion (UNIX epoch milliseconds → datetime)
  - Deduplication (exact row-level DISTINCT)
  - Outlier Removal (Filters out bot-like visitors exceeding the 99.9th percentile of event counts)
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import duckdb

# =============================================================================
# Public Functions
# =============================================================================


def clean_data(df_input):
    """
    Clean a raw event DataFrame using DuckDB SQL.

    This function performs four operations in a single SQL query:
      1. Fills missing values:
         - 'transactionid' → filled with 0
         - Numeric columns  → filled with -1
         - String columns   → filled with '-1'
      2. Converts timestamp columns from UNIX epoch milliseconds to DuckDB TIMESTAMP.
      3. Removes exact duplicate rows using SELECT DISTINCT.
      4. Removes event frequency outliers (e.g., potential bots/crawlers).
    """
    orig_count = duckdb.sql("SELECT COUNT(*) FROM df_input").fetchone()[0]
    print(f"Original shape: ({orig_count}, {df_input.shape[1]})")

    select_exprs = []
    for col in df_input.columns:
        if col == 'transactionid':
            select_exprs.append(f"COALESCE({col}, 0) AS {col}")
        elif 'timestamp' in col.lower():
            select_exprs.append(f"epoch_ms({col}) AS {col}")
        else:
            if pd.api.types.is_numeric_dtype(df_input[col]):
                select_exprs.append(f"COALESCE({col}, -1) AS {col}")
            else:
                select_exprs.append(f"COALESCE({col}, '-1') AS {col}")

    select_clause = ",\n        ".join(select_exprs)

    # ------------------------------------------------------------------
    # Step 1: DISTINCT and Imputation (Base CTE)
    # Step 2: Calculate event counts per visitor
    # Step 3: Determine the 99.9th percentile threshold to identify bots
    # Step 4: Filter out those extreme outliers
    # ------------------------------------------------------------------
    query = f"""
    WITH base_cleaned AS (
        SELECT DISTINCT
            {select_clause}
        FROM df_input
    ),
    visitor_counts AS (
        SELECT 
            visitorid, 
            COUNT(*) as event_count
        FROM base_cleaned
        GROUP BY visitorid
    ),
    threshold AS (
        -- Calculate the 99.9% percentile of event frequencies
        SELECT quantile_cont(event_count, 0.999) as max_allowed_events 
        FROM visitor_counts
    )
    SELECT b.*
    FROM base_cleaned b
    JOIN visitor_counts v ON b.visitorid = v.visitorid
    CROSS JOIN threshold t
    WHERE v.event_count <= t.max_allowed_events
    """

    cleaned_df = duckdb.sql(query).df()

    print(f"Cleaned shape (Outliers & Duplicates removed): ({cleaned_df.shape[0]}, {cleaned_df.shape[1]})")
    print("=" * 30)

    return cleaned_df