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
    df_copy["Cluster"] = labels

    # Calculate the mean of each RFM metric per cluster
    profiles = (
        df_copy.groupby("Cluster")
        .agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"})
        .round(2)
    )

    # Add the number of customers in each cluster
    profiles["Count"] = df_copy.groupby("Cluster")["Recency"].count()

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
    """


def get_business_strategies(profiles, baseline_stats):
    """
    Map cluster profiles to business personas and actionable strategies.

    This function implements a refined heuristic to assign names and marketing
    strategies to segments. It differentiates between various types of low
    engagement (e.g., new vs. dormant) to provide more actionable insights.

    Args:
        profiles (pd.DataFrame): Output of get_cluster_profiles(), indexed
            by 'Cluster' with columns ['Recency', 'Frequency', 'Monetary', 'Count'].
        baseline_stats (dict, optional): Dictionary with global RFM means/medians
            to use as thresholds. If None, uses the mean of the cluster profiles.

    Returns:
        pd.DataFrame: A DataFrame indexed by 'Cluster' with columns
            ['Persona', 'Strategy'].
    """
    strategies = []

    # Calculate thresholds based on baseline or cluster averages
    if baseline_stats:
        avg_r = baseline_stats.get("Recency", profiles["Recency"].mean())
        avg_f = baseline_stats.get("Frequency", profiles["Frequency"].mean())
        avg_m = baseline_stats.get("Monetary", profiles["Monetary"].mean())
    else:
        avg_r = profiles["Recency"].mean()
        avg_f = profiles["Frequency"].mean()
        avg_m = profiles["Monetary"].mean()

    for cluster_id, row in profiles.iterrows():
        # ------------------------------------------------------------------
        # Rule 0: Noise cluster from DBSCAN (label = -1).
        # ------------------------------------------------------------------
        if cluster_id == -1:
            persona = "Outliers/Noise"
            strategy = "Ignore or investigate for unusual behavior (bots/crawlers)."

        else:
            # ------------------------------------------------------------------
            # Tier 1: High Engagement
            # ------------------------------------------------------------------
            if row["Frequency"] > avg_f:
                if row["Recency"] <= avg_r:
                    persona = "Loyal Customers"
                    strategy = (
                        "Reward with exclusive offers; Early access to new items."
                    )
                else:
                    persona = "At Risk / Churning"
                    strategy = "Send re-engagement emails; Discount codes for return."

            # ------------------------------------------------------------------
            # Tier 2: Low Engagement - Differentiated by Recency
            # ------------------------------------------------------------------
            elif row["Frequency"] <= avg_f and row["Monetary"] <= avg_m:
                if row["Recency"] <= avg_r:
                    persona = "New / Recent Browsers"
                    strategy = "Awareness campaigns; Retarget with trending items."
                else:
                    persona = "Dormant Leads"
                    strategy = "Win-back discounts; Newsletter reminders."

            # ------------------------------------------------------------------
            # Tier 3: Selective Engagement
            # ------------------------------------------------------------------
            elif row["Monetary"] > avg_m:
                persona = "Occasional High-Value"
                strategy = "Upsell related categories; Incentive for higher frequency."

            # ------------------------------------------------------------------
            # Tier 4: Default / Catch-all
            # ------------------------------------------------------------------
            else:
                persona = "Potential Loyalists"
                strategy = "Invite to loyalty program; Personalize recommendations."

        strategies.append(
            {"Cluster": cluster_id, "Persona": persona, "Strategy": strategy}
        )

    return pd.DataFrame(strategies).set_index("Cluster")


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
    sns.histplot(rfm_df["Recency"], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Recency Distribution")

    # Frequency distribution (typically heavily right-skewed)
    sns.histplot(rfm_df["Frequency"], kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Frequency Distribution")

    # Monetary distribution (typically heavily right-skewed, correlated with Frequency)
    sns.histplot(rfm_df["Monetary"], kde=True, ax=axes[2], color="lightgreen")
    axes[2].set_title("Monetary Distribution")

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

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("RFM Correlation Heatmap", fontsize=14)
    plt.show()
