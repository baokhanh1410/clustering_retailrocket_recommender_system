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
- Handling missing values and duplicates.
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
| View | 1 point |
| Add-to-cart | 3 points |
| Transaction | 5 points |

### Feature Scaling

Data normalization is performed using `StandardScaler` to ensure balanced feature importance before clustering.

---

### 3. Clustering Models

The project implements and compares multiple clustering algorithms:

#### 🔹 K-Means
- Primary baseline clustering model.
- Centroid-based segmentation approach.

#### 🔹 DBSCAN
- Density-based clustering algorithm.
- Effective for discovering arbitrary-shaped clusters and handling noise.

#### 🔹 Hierarchical Clustering
- Builds nested cluster structures.
- Visualized using Dendrograms for deeper structural understanding.

---

### 4. Model Evaluation

| Method | Purpose |
|---|---|
| **Elbow Method** | Determine the optimal number of clusters (`k`) |
| **Silhouette Score** | Evaluate cluster separation quality and density |

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
git clone https://github.com/baokhanh1410/clustering_retailrocket_recommender_system-.git
cd DA_Classification
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
|---|---|
| **Loyal Customers** | High frequency and recent activity |
| **At-Risk Users** | Previously active users with long inactivity |
| **Window Shoppers** | Frequent views but no transactions |

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
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📂 Project Structure

```bash
DA_Classification/
│
├── dataset/                  # Dataset storage (ignored in Git)
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   └── cleaning_data.py
│   └── data_integration.py
│   └── cleaning_data.py
│   └── cleaning_data.py
├── main.ipynb             # Main analysis notebook
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
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