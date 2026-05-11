"""
cleaning_data.py — Data Cleaning Module via DuckDB SQL

This module provides a function to clean a raw pandas DataFrame using DuckDB's
in-process SQL engine.  The cleaning pipeline handles:
  - Missing value imputation (NaN → 0 for transactionid, -1 for others)
  - Timestamp conversion (UNIX epoch milliseconds → datetime)
  - Deduplication (exact row-level DISTINCT)

DuckDB is used instead of pandas for cleaning because it can process large
DataFrames more efficiently using columnar execution and SQL semantics.
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

    This function performs three operations in a single SQL query:
      1. Fills missing values:
         - 'transactionid' → filled with 0 (customers without purchases)
         - Numeric columns  → filled with -1 (unknown/unavailable)
         - String columns   → filled with '-1' (unknown/unavailable)
      2. Converts timestamp columns from UNIX epoch milliseconds to
         DuckDB TIMESTAMP type using the epoch_ms() function.
      3. Removes exact duplicate rows using SELECT DISTINCT.

    The function prints the original and cleaned shapes for quick
    verification of how many rows were removed.

    Args:
        df_input (pd.DataFrame): Raw DataFrame to clean.  Expected to
            contain columns like 'timestamp', 'visitorid', 'event',
            'itemid', 'transactionid', etc.

    Returns:
        pd.DataFrame: Cleaned DataFrame with missing values filled,
            timestamps converted, and duplicates removed.
    """
    orig_count = duckdb.sql("SELECT COUNT(*) FROM df_input").fetchone()[0]
    print(f"Original shape: ({orig_count}, {df_input.shape[1]})")

    # ------------------------------------------------------------------
    # Build the SELECT expression list dynamically for each column.
    #
    # Each column gets a tailored COALESCE expression:
    #   - 'transactionid': fill NaN with 0 because customers who haven't
    #     purchased have no transaction ID, but we want a numeric placeholder.
    #   - Timestamp columns: convert from UNIX epoch (ms) to TIMESTAMP
    #     using DuckDB's epoch_ms() function.
    #   - Other numeric columns: fill NaN with -1 (sentinel for "unknown").
    #   - Other string columns: fill NaN with '-1' (string sentinel).
    # ------------------------------------------------------------------
    select_exprs = []
    for col in df_input.columns:
        if col == 'transactionid':
            # Fill missing transaction IDs with 0 (no purchase)
            select_exprs.append(f"COALESCE({col}, 0) AS {col}")

        elif 'timestamp' in col.lower():
            # Convert UNIX epoch milliseconds to a proper TIMESTAMP type
            select_exprs.append(f"epoch_ms({col}) AS {col}")

        else:
            # For all other columns, fill NaN based on data type
            if pd.api.types.is_numeric_dtype(df_input[col]):
                # Numeric columns: fill with -1 (sentinel value)
                select_exprs.append(f"COALESCE({col}, -1) AS {col}")
            else:
                # String/object columns: fill with '-1' (string sentinel)
                select_exprs.append(f"COALESCE({col}, '-1') AS {col}")

    select_clause = ",\n        ".join(select_exprs)

    # ------------------------------------------------------------------
    # The DISTINCT keyword removes exact duplicate rows.
    # This is equivalent to pandas' drop_duplicates() but executed
    # inside DuckDB for better performance on large datasets.
    # ------------------------------------------------------------------
    query = f"""
    SELECT DISTINCT
        {select_clause}
    FROM df_input
    """

    # Execute the cleaning query and materialize as a pandas DataFrame
    cleaned_df = duckdb.sql(query).df()

    # Report the cleaned shape using the ACTUAL result dimensions
    print(f"Cleaned shape: ({cleaned_df.shape[0]}, {cleaned_df.shape[1]})")
    print("=" * 30)

    return cleaned_df