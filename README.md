# Clustering: Retailrocket Recommender System & Customer Segmentation

## 👤 Student Information

| Name | Student ID |
|---|---|
| Nguyễn Bảo Khánh | 11247178 |
| Trần Quang Huy | 11247176 |
| Nguyễn Hoàng Thành | 11247224 |
| Đào Phát Nhật | 11247123 |

### Academic Information

| Field | Information |
|---|---|
| **Class** | DS66B |
| **Major** | Data Science |
| **Department** | Faculty of Data Science and AI (FDA) |
| **University** | National Economics University (NEU) |
| **Subject** | Data Analysis with Python |
| **Instructor** | Trần Đức Minh |

---

## 📌 Project Overview

This project focuses on **Unsupervised Learning** to perform **Customer Segmentation** for the Retailrocket eCommerce platform.  
By analyzing user behavior without pre-defined labels, the system identifies natural groupings of visitors to enhance personalized marketing and recommendation strategies.

---

## 🎯 Objectives

- Perform customer segmentation based on shopping behaviors.
- Identify distinct user profiles using unsupervised clustering algorithms.
- Provide actionable business insights for different customer tiers.
- Compare multiple clustering techniques for behavioral analysis.

---

## 📊 Dataset

This project utilizes the **Retailrocket eCommerce Dataset** from Kaggle, which includes:

| File | Description |
|---|---|
| `events.csv` | User interactions (`view`, `addtocart`, `transaction`) |
| `category_tree.csv` | Hierarchy of product categories |
| `item_properties.csv` | Attributes of items (distributed across multiple files) |

> **Note:**  
> Due to the large dataset size (~2GB), raw data files are **not included** in this repository.  
> The dataset is automatically downloaded using the Kaggle API.

---

## 🛠 Technical Pipeline

### 1. Data Integration & Cleaning

- Merging item properties and category structures into interaction logs using:
  - `visitorid`
  - `itemid`
- Handling missing values, duplicates and outliers.
- Converting UNIX timestamps into human-readable datetime formats.

---

### 2. Feature Engineering (RFM Analysis)

A customized **RFM (Recency, Frequency, Monetary)** framework is created for each visitor:

| Metric | Description |
|---|---|
| **Recency (R)** | Days since the last interaction |
| **Frequency (F)** | Total number of user interactions |
| **Monetary / Engagement (M)** | Engagement score based on activity type |

### Engagement Scoring

| Event Type | Score |
|---|---|
| View | 0 point |
| Add-to-cart | 30 points |
| Transaction | 50 points |

### Feature Scaling

Data normalization is performed using `StandardScaler` to ensure balanced feature importance before clustering.

---

### 3. Clustering Models

The project implements and compares multiple clustering algorithms:

#### 🔹 K-Means
- Primary baseline clustering model.
- Centroid-based segmentation approach optimizing within-cluster variance.
- Guided by the **Elbow Method** for optimal K selection.

#### 🔹 DBSCAN
- Density-based clustering algorithm.
- Effective for discovering arbitrary-shaped clusters and automatically identifying **noise/outlier points** (labeled as -1).
- Parameters `eps` and `min_samples` are tuned to find optimal data density.

#### 🔹 Hierarchical Clustering
- Agglomerative (bottom-up) approach using **Ward linkage**.
- Provides a deterministic cluster structure visualized using **Dendrograms** to guide the selection of cluster counts.
- Useful for understanding the taxonomic relationships between customer segments.

---

### 4. Model Comparison & Evaluation

| Method | Metric | Purpose |
|---|---|---|
| **K-Means** | Inertia / Silhouette | Centroid cohesion and cluster separation |
| **DBSCAN** | Silhouette Score | Identification of high-density regions vs. noise |
| **Hierarchical** | Cophenetic / Silhouette | Visualizing merge distances via dendrogram |
| **Cross-Model** | Silhouette Score | Side-by-side comparison of all three paradigms |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Kaggle API Key (`kaggle.json`) configured in:

```bash
~/.kaggle/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/baokhanh1410/clustering_retailrocket_recommender_system.git
cd clustering_retailrocket_recommender_system
```

### 2. Create and Activate Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation
#### Method 1: Use Kaggle API (Automatic)
The project integrates the dataset.py module to automatically download data. To use:

Login Kaggle, go to Settings -> Create New API Token to download the kaggle.json file.

Move the kaggle.json file to the folder:

- **Windows**: `C:\Users\<Username>\.kaggle\`
- **Linux/macOS**: `~/.kaggle/`

#### Method 2: Manual Download

Access: [Retailrocket eCommerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Download, unzip and copy all .csv files into the dataset/ folder in the project root directory.

---

## ▶️ Running the Project

Open the notebook:

```bash
main.ipynb
```

The notebook is configured to:

- Automatically download the dataset via Kaggle API.
- Perform the complete data pipeline.
- Execute clustering models.
- Generate customer segmentation insights and visualizations.

---

## 💡 Insights & Customer Profiling

Each cluster is profiled using average **RFM** scores:

| Customer Type | Characteristics |
| :--- | :--- |
| **Loyal Customers** | High-value users with high **Cumulative Engagement Score**, driven by multiple cart additions or transactions. |
| **New / Recent Browsers** | Users who interacted with the platform very recently and are currently in the discovery phase. |
| **At-Risk Users** | Previously active users with moderate frequency |
| **Dormant Leads** | The largest segment, characterized by high inactivity and zero engagement score beyond casual browsing. |

These insights can help businesses:

- Improve targeted marketing campaigns.
- Increase customer retention.
- Personalize recommendation systems.
- Optimize conversion strategies.

---

## 📈 Technologies Used

- Python
- Pandas
- NumPy
- Scipy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Duckdb

---

## 📂 Project Structure

```bash
clustering_retailrocket_recommender_system/
│
├── dataset/                  # Dataset storage (ignored in Git)
├── src/
│   ├── __init__.py
│   ├── analysis.py            # Visualization & strategy mapping
│   ├── cleaning_data.py      # Data cleaning logic
│   ├── data_integration.py   # DuckDB merge logic
│   ├── dataset.py            # Kaggle data setup
│   ├── features.py           # RFM engineering & scaling
│   └── models.py             # Clustering algorithms
├── main.ipynb                 # Main analysis notebook
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## 📜 License

This project is developed for educational purposes as part of the **Data Analysis with Python** course at **National Economics University (NEU)**.

---

## 🙌 Acknowledgements

- Retailrocket for providing the dataset.
- Kaggle platform for dataset hosting.
- Faculty of Data Science and AI (FDA), NEU.
- Instructor: **Trần Đức Minh**
