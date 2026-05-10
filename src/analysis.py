import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_cluster_profiles(rfm_df, labels):
    """
    Calculates mean RFM values for each cluster.
    """
    df_copy = rfm_df.copy()
    df_copy['Cluster'] = labels
    
    # Calculate mean for each cluster
    profiles = df_copy.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    
    # Add cluster size
    profiles['Count'] = df_copy.groupby('Cluster')['Recency'].count()
    
    return profiles

def get_business_strategies(profiles):
    """
    Maps cluster characteristics to personas and strategies.
    This is a rule-based heuristic for the Retailrocket dataset.
    """
    strategies = []
    
    for cluster_id, row in profiles.iterrows():
        if cluster_id == -1:
            persona = "Outliers/Noise"
            strategy = "Ignore or investigate for unusual behavior (bots/crawlers)."
        else:
            # Logic based on relative RFM
            # (In a real scenario, we would compare against global means)
            if row['Frequency'] > profiles['Frequency'].mean() and row['Recency'] < profiles['Recency'].mean():
                persona = "Loyal Customers"
                strategy = "Reward with exclusive offers; Early access to new items."
            elif row['Recency'] > profiles['Recency'].mean() and row['Frequency'] > profiles['Frequency'].mean():
                persona = "At Risk / Churning"
                strategy = "Send re-engagement emails; Discount codes for return."
            elif row['Frequency'] < profiles['Frequency'].mean() and row['Monetary'] < profiles['Monetary'].mean():
                persona = "Window Shoppers"
                strategy = "Retarget with popular items; Awareness campaigns."
            else:
                persona = "Potential Loyalists"
                strategy = "Upsell related categories; Loyalty program invitations."
                
        strategies.append({
            'Cluster': cluster_id,
            'Persona': persona,
            'Strategy': strategy
        })
        
    return pd.DataFrame(strategies).set_index('Cluster')

def plot_rfm_distributions(rfm_df, title="RFM Distributions"):
    """
    Plots histograms for Recency, Frequency, and Monetary.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(rfm_df['Recency'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Recency Distribution')
    
    sns.histplot(rfm_df['Frequency'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Frequency Distribution')
    
    sns.histplot(rfm_df['Monetary'], kde=True, ax=axes[2], color='lightgreen')
    axes[2].set_title('Monetary Distribution')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_rfm_correlation(rfm_df):
    """
    Plots a heatmap of the correlation matrix for RFM variables.
    """
    plt.figure(figsize=(8, 6))
    corr = rfm_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('RFM Correlation Heatmap', fontsize=14)
    plt.show()
