<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🛒%20SmartCart%20Customer%20Segmentation&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Unsupervised%20Learning%20with%20K-Means%20%26%20Agglomerative%20Clustering%20%2B%20PCA&descAlignY=60&descAlign=50" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

SmartCart Analytics is an **interactive Streamlit dashboard** that transforms raw customer data into actionable marketing intelligence using unsupervised machine learning. The app applies **K-Means** and **Agglomerative Clustering** on PCA-reduced features to group customers into meaningful segments — then auto-generates tailored **marketing strategies** for each group.

- **Task:** Unsupervised Customer Segmentation
- **Dataset:** Cleaned retail customer marketing dataset (`df_cleaned.csv`)
- **Goal:** Identify distinct customer segments and recommend data-driven marketing actions per segment

---

## 📂 Dataset

| Property | Details |
|:---|:---|
| Source | `df_cleaned.csv` (pre-processed retail marketing data) |
| Total Samples | ~2,200 customers |
| Features Used | 13 (`Income`, `Age`, `Total_Spending`, `Recency`, `NumWebPurchases`, `NumStorePurchases`, `NumCatalogPurchases`, `NumDealsPurchases`, `NumWebVisitsMonth`, `Customer_Tenure_Days`, `Total_Children`, `Education`, `Living_With`) |
| Target | No label — unsupervised clustering |
| Outliers Removed | Age ≥ 90 and Income ≥ 6,00,000 excluded |

---

## 🔄 Pipeline Workflow

```
Raw Data → Outlier Removal → One-Hot Encoding → Standard Scaling → PCA (3D) → Clustering → Profiling → Strategy
```

1️⃣ **Data Loading** — Customer dataset loaded from `df_cleaned.csv`

2️⃣ **Outlier Removal** — Customers with Age ≥ 90 or Income ≥ 6,00,000 filtered out

3️⃣ **Encoding** — `Education` and `Living_With` one-hot encoded via `OneHotEncoder`

4️⃣ **Scaling** — All features standardised using `StandardScaler`

5️⃣ **Dimensionality Reduction** — PCA applied to reduce to 3 principal components

6️⃣ **Clustering** — K-Means and Agglomerative (Ward linkage) run on PCA space; K selectable via sidebar slider (2–8)

7️⃣ **Evaluation** — Elbow curve (WCSS) and Silhouette scores computed across K = 2–10

8️⃣ **Profiling & Strategy** — Per-segment radar charts, feature comparisons, and auto-generated marketing recommendations

---

## 🤖 Models

### 1️⃣ K-Means Clustering

```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)
```

- Centroid-based clustering on PCA-reduced space
- Fast and scalable for large datasets
- Optimal K determined via Elbow + Silhouette analysis
- Default: **K = 4 clusters**

---

### 2️⃣ Agglomerative Clustering ⭐ Also Supported

```python
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels = agg.fit_predict(X_pca)
```

- Hierarchy-based, no centroid assumption
- Ward linkage minimises within-cluster variance
- Useful for detecting non-spherical cluster shapes
- Produces visually similar separation to K-Means in PCA space

---

## 📊 Dashboard Tabs

| Tab | What You Get |
|:---|:---|
| 📊 **Overview** | KPI cards, cluster distribution bar chart, Income vs Spending scatter, Education & Living situation donut charts, PCA variance explained |
| 🔬 **EDA** | Feature distributions, correlation heatmap, and pairwise relationship plots |
| 📈 **Optimal K** | Elbow curve (WCSS) and Silhouette score chart across K = 2–10 |
| 🗺️ **Clusters 3D** | Interactive 3D PCA scatter, cluster size summary, Silhouette score, K-Means vs Agglomerative comparison |
| 👥 **Segment Profiles** | Radar chart, feature-mean bar chart per cluster, full colour-coded summary table |
| 🎯 **Strategy** | Auto-generated marketing recommendations per segment + CSV export of segmented data |

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| Default Clusters (K) | 4 |
| PCA Variance Captured | ~70–80% (3 components) |
| Silhouette Score (K=4) | ~0.28–0.35 |
| Algorithms Supported | K-Means · Agglomerative (Ward) |

---

## 🔍 Key Insights

- 🧠 **PCA with 3 components** captures the majority of variance while enabling 3D visualisation of cluster separation
- 💎 **High Earners** show strong preference for web and catalog purchases — ideal for digital retargeting campaigns
- 🏷️ **Value Seekers** are heavily deal-driven — flash sales and loyalty discounts maximise their response rate
- 👶 **Family segments** (higher `Total_Children`) respond best to bundle offers and back-to-school promotions
- ⚖️ **K-Means vs Agglomerative** produce similar cluster shapes in PCA space — K-Means is preferred for speed at scale
- 🎯 Segment-aware marketing can significantly improve campaign ROI by targeting the right channel per cluster

---

## 🗂️ Repository Structure

```
SmartCart-Customer-Segmentation/
│
├── smartcart_app.py        # Main Streamlit application
├── df_cleaned.csv          # Cleaned customer dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/SmartCart-Customer-Segmentation.git
cd SmartCart-Customer-Segmentation

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run smartcart_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 🧠 Key Learnings

- Unsupervised learning requires careful preprocessing — scaling and encoding directly impact cluster quality
- PCA is essential for both dimensionality reduction and enabling 3D cluster visualisation
- Silhouette score and Elbow curve together provide a more reliable K selection than either alone
- Agglomerative clustering (Ward) and K-Means converge to similar results on compact, well-scaled data
- Segment-level profiling is more actionable than raw cluster labels — naming and interpreting clusters drives real business value

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| Streamlit | Interactive web dashboard |
| Pandas | Data manipulation & filtering |
| scikit-learn | KMeans, Agglomerative, PCA, StandardScaler, Silhouette |
| Plotly | Interactive charts, 3D scatter, radar plots |
| Seaborn / Matplotlib | Supplementary EDA visualisations |
| NumPy | Numerical computation |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronakrajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
