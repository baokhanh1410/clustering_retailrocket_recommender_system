"""
analysis.py — Cluster Profiling, Business Strategy Mapping & RFM Visualization

This module provides functions for analyzing clustering results and generating
actionable business insights from RFM (Recency, Frequency, Monetary) data:

  - get_cluster_profiles():    Computes mean RFM values and size for each cluster.
  - get_business_strategies(): Maps cluster profiles to customer personas and
                               recommends marketing strategies using rule-based heuristics.
  - plot_rfm_distributions():  Plots histograms for each RFM dimension.
  - plot_rfm_correlation():    Plots a correlation heatmap for RFM features.
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Cluster Profiling Functions
# =============================================================================


def get_cluster_profiles(rfm_df, labels):
    """
    Calculate mean RFM values and cluster sizes for each cluster.

    Creates a profile table that summarizes the average Recency, Frequency,
    and Monetary values for each cluster, along with the number of customers
    in each cluster.  This is the primary input for persona assignment.

    Args:
        rfm_df (pd.DataFrame): Original (unscaled) RFM DataFrame with columns
            'Recency', 'Frequency', 'Monetary'.
        labels (array-like): Cluster labels assigned to each row in rfm_df.
            May include -1 for noise points (from DBSCAN).

    Returns:
        pd.DataFrame: A DataFrame indexed by 'Cluster' with columns
            ['Recency', 'Frequency', 'Monetary', 'Count'], all values
            rounded to 2 decimal places.
    """
    df_copy = rfm_df.copy()
    df_copy['Cluster'] = labels

    # Calculate the mean of each RFM metric per cluster
    profiles = df_copy.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)

    # Add the number of customers in each cluster
    profiles['Count'] = df_copy.groupby('Cluster')['Recency'].count()

    return profiles


# =============================================================================
# Business Strategy Functions
# =============================================================================


def get_business_strategies(profiles):
    """
    Map cluster profiles to customer personas and marketing strategies.

    Uses a rule-based heuristic that compares each cluster's RFM averages
    against the overall mean across all clusters.  The classification logic
    follows standard RFM interpretation:

      - HIGH Recency  = visitor has been INACTIVE for a long time (bad).
      - LOW  Recency  = visitor was recently active (good).
      - HIGH Frequency = visitor interacts often (good).
      - LOW  Frequency = visitor rarely interacts (bad).
      - HIGH Monetary  = visitor has high engagement score (good).
      - LOW  Monetary  = visitor has low engagement (bad).

    Persona assignment rules:
      1. Cluster -1 (DBSCAN noise) → "Outliers/Noise" — ignore or investigate.
      2. High Frequency AND Low Recency → "Loyal Customers" — recently active
         and frequently engaged.  Strategy: reward with exclusive offers.
      3. High Recency AND High Frequency → "At Risk / Churning" — used to be
         active but haven't visited recently.  Strategy: re-engagement campaigns.
      4. Low Frequency AND Low Monetary → "Window Shoppers" — infrequent,
         low-value interactions.  Strategy: retarget with popular items.
      5. Everything else → "Potential Loyalists" — moderate engagement that
         could be nurtured.  Strategy: upsell and loyalty programs.

    Args:
        profiles (pd.DataFrame): Output of get_cluster_profiles(), indexed
            by 'Cluster' with columns ['Recency', 'Frequency', 'Monetary', 'Count'].

    Returns:
        pd.DataFrame: A DataFrame indexed by 'Cluster' with columns
            ['Persona', 'Strategy'].
    """
    strategies = []

    for cluster_id, row in profiles.iterrows():

        # ------------------------------------------------------------------
        # Rule 0: Noise cluster from DBSCAN (label = -1).
        # These points didn't fit into any cluster and may represent
        # bots, crawlers, or genuinely unusual behavior.
        # ------------------------------------------------------------------
        if cluster_id == -1:
            persona = "Outliers/Noise"
            strategy = "Ignore or investigate for unusual behavior (bots/crawlers)."

        else:
            # ------------------------------------------------------------------
            # Rule 1: Loyal Customers
            #   - Frequency ABOVE average → actively engaged
            #   - Recency BELOW average  → recently visited (still active)
            # These are the most valuable customers.
            # ------------------------------------------------------------------
            if row['Frequency'] > profiles['Frequency'].mean() and row['Recency'] < profiles['Recency'].mean():
                persona = "Loyal Customers"
                strategy = "Reward with exclusive offers; Early access to new items."

            # ------------------------------------------------------------------
            # Rule 2: At Risk / Churning
            #   - Recency ABOVE average  → haven't visited in a while
            #   - Frequency ABOVE average → USED TO be active
            # These customers were once loyal but are drifting away.
            # ------------------------------------------------------------------
            elif row['Recency'] > profiles['Recency'].mean() and row['Frequency'] > profiles['Frequency'].mean():
                persona = "At Risk / Churning"
                strategy = "Send re-engagement emails; Discount codes for return."

            # ------------------------------------------------------------------
            # Rule 3: Window Shoppers
            #   - Frequency BELOW average → rarely interact
            #   - Monetary BELOW average  → low engagement value
            # These customers browse but don't convert.
            # ------------------------------------------------------------------
            elif row['Frequency'] < profiles['Frequency'].mean() and row['Monetary'] < profiles['Monetary'].mean():
                persona = "Window Shoppers"
                strategy = "Retarget with popular items; Awareness campaigns."

            # ------------------------------------------------------------------
            # Rule 4: Potential Loyalists (default / catch-all)
            #   - Don't fit the above patterns clearly.
            #   - Show moderate engagement that could be improved.
            # ------------------------------------------------------------------
            else:
                persona = "Potential Loyalists"
                strategy = "Upsell related categories; Loyalty program invitations."

        strategies.append({
            'Cluster': cluster_id,
            'Persona': persona,
            'Strategy': strategy
        })

    return pd.DataFrame(strategies).set_index('Cluster')


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_rfm_distributions(rfm_df, title="RFM Distributions"):
    """
    Plot histograms with KDE curves for Recency, Frequency, and Monetary.

    Displays three side-by-side histograms to visualize the distribution
    of each RFM metric.  This is useful for identifying skewness and
    deciding whether log transformation is needed before clustering.

    Args:
        rfm_df (pd.DataFrame): RFM DataFrame with columns
            'Recency', 'Frequency', 'Monetary'.
        title (str): Overall figure title. Defaults to "RFM Distributions".

    Returns:
        None: Displays the plot inline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Recency distribution (typically more uniform)
    sns.histplot(rfm_df['Recency'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Recency Distribution')

    # Frequency distribution (typically heavily right-skewed)
    sns.histplot(rfm_df['Frequency'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Frequency Distribution')

    # Monetary distribution (typically heavily right-skewed, correlated with Frequency)
    sns.histplot(rfm_df['Monetary'], kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_title('Monetary Distribution')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_rfm_correlation(rfm_df):
    """
    Plot a heatmap of the correlation matrix for RFM variables.

    Helps identify multicollinearity between features.  In this dataset,
    Frequency and Monetary are expected to be highly correlated because
    Monetary is derived from the same events that determine Frequency.

    Args:
        rfm_df (pd.DataFrame): RFM DataFrame with numeric columns.

    Returns:
        None: Displays the heatmap inline.
    """
    plt.figure(figsize=(8, 6))

    # Use numeric_only=True to avoid FutureWarning in pandas >= 1.5
    # when the DataFrame may contain non-numeric columns.
    corr = rfm_df.corr(numeric_only=True)

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('RFM Correlation Heatmap', fontsize=14)
    plt.show()
