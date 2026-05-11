"""
data_integration.py — Multi-Source Data Loading & Merging Module

This module provides functions for loading CSV files into DuckDB relations
and merging multiple data sources (events, item properties, category tree)
into a single unified DataFrame using DuckDB SQL joins.

The merge strategy uses a CTE (Common Table Expression) to:
  1. Union the two item_properties parts.
  2. Extract the latest categoryid for each item using QUALIFY + ROW_NUMBER.
  3. LEFT JOIN the category information onto the events table.

Key Exports:
    - load_data(path):   Reads a CSV file into a DuckDB relation.
    - merge_data(data):  Merges events, item properties, and category tree.
"""

# =============================================================================
# Imports
# =============================================================================

import duckdb


# =============================================================================
# Public Functions
# =============================================================================


def load_data(path):
    """
    Load a CSV file into a DuckDB relation.

    DuckDB relations are lazy — they don't load the entire file into memory
    immediately.  This makes them efficient for large datasets.

    Args:
        path (str): File path to the CSV file.

    Returns:
        duckdb.DuckDBPyRelation: A DuckDB relation representing the CSV data.
    """
    return duckdb.read_csv(path)


def merge_data(data):
    """
    Merge events, item properties, and category tree into a single DataFrame.

    Takes a list of four DuckDB relations (in order) and joins them:
      1. events              — main interaction log (visitorid, itemid, event, ...)
      2. item_properties_1   — first part of item properties (timestamp, itemid, property, value)
      3. item_properties_2   — second part of item properties (same schema)
      4. category_tree       — category hierarchy (categoryid, parentid)

    The merge strategy:
      - UNION ALL the two item_properties parts.
      - Filter for rows where property = 'categoryid' to extract category info.
      - Use QUALIFY ROW_NUMBER() to keep only the LATEST categoryid per item
        (partitioned by itemid, ordered by timestamp DESC).
      - LEFT JOIN this category mapping onto the events table.
      - If category_tree has a 'categoryid' column, also LEFT JOIN the tree
        to bring in the parent category hierarchy.

    Args:
        data (list): A list of exactly 4 DuckDB relations in this order:
            [events, item_properties_1, item_properties_2, category_tree].

    Returns:
        pd.DataFrame: A merged DataFrame with event data enriched by
            category information.
    """
    events = data[0]
    item_properties_1 = data[1]
    item_properties_2 = data[2]
    category_tree = data[3]

    # ------------------------------------------------------------------
    # Check if the category_tree relation contains a 'categoryid' column.
    # DuckDB relations expose a .columns property that returns column names.
    # We convert to a list for safe membership testing, which works for
    # both DuckDB relations and pandas DataFrames.
    # ------------------------------------------------------------------
    available_columns = list(category_tree.columns)
    has_categoryid = 'categoryid' in available_columns

    if has_categoryid:
        # ------------------------------------------------------------------
        # Full merge: events ← item_categories ← category_tree
        #
        # CTE `item_categories`:
        #   - UNION ALL merges both item_properties parts.
        #   - Filters for property = 'categoryid' to find category assignments.
        #   - QUALIFY ROW_NUMBER() keeps only the LATEST category per item
        #     (items can change categories over time; we want the most recent).
        #
        # Main query:
        #   - LEFT JOIN item_categories to get each event's item category.
        #   - LEFT JOIN category_tree to get the parent category hierarchy.
        #   - EXCLUDE (categoryid) on category_tree avoids duplicate column
        #     names in the result when both joins produce a 'categoryid'.
        # ------------------------------------------------------------------
        query = """
            WITH item_categories AS (
                SELECT 
                    CAST(itemid AS VARCHAR) AS itemid, 
                    CAST(value AS VARCHAR) AS categoryid
                FROM (
                    SELECT * FROM item_properties_1
                    UNION ALL 
                    SELECT * FROM item_properties_2
                )
                WHERE property = 'categoryid'
                QUALIFY ROW_NUMBER() OVER(PARTITION BY itemid ORDER BY timestamp DESC) = 1
            )
            SELECT 
                e.*,
                c.categoryid,
                t.* EXCLUDE (categoryid)
            FROM events e
            LEFT JOIN item_categories c 
                ON CAST(e.itemid AS VARCHAR) = c.itemid
            LEFT JOIN category_tree t 
                ON c.categoryid = CAST(t.categoryid AS VARCHAR)
        """
    else:
        # ------------------------------------------------------------------
        # Partial merge: events ← item_categories only
        #
        # If category_tree doesn't have 'categoryid' (unexpected schema),
        # we still extract categories from item_properties but skip the
        # tree join.  This gracefully handles missing hierarchy data.
        # ------------------------------------------------------------------
        query = """
            WITH item_categories AS (
                SELECT 
                    CAST(itemid AS VARCHAR) AS itemid, 
                    CAST(value AS VARCHAR) AS categoryid
                FROM (
                    SELECT * FROM item_properties_1
                    UNION ALL 
                    SELECT * FROM item_properties_2
                )
                WHERE property = 'categoryid'
                QUALIFY ROW_NUMBER() OVER(PARTITION BY itemid ORDER BY timestamp DESC) = 1
            )
            SELECT 
                e.*,
                c.categoryid
            FROM events e
            LEFT JOIN item_categories c 
                ON CAST(e.itemid AS VARCHAR) = c.itemid
        """

    return duckdb.sql(query).df()