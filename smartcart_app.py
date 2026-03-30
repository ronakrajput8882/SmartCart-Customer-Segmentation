import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCart · Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f1a;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12141f !important;
    border-right: 1px solid #1e2035;
}
[data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
    font-family: 'Syne', sans-serif;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #7c7f9e;
    margin-bottom: 1.5rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e0e0f5;
    border-left: 4px solid #a78bfa;
    padding-left: 0.8rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1a1d2e, #1e2035);
    border: 1px solid #2a2d42;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
}
.kpi-label {
    font-size: 0.78rem;
    color: #7c7f9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Cluster badge */
.cluster-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #12141f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #7c7f9e;
    border-radius: 8px;
    padding: 0.5rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: #1e2035 !important;
    color: #a78bfa !important;
}

/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.27), rgba(56,189,248,0.27), transparent);
    margin: 1.5rem 0;
}

/* Insight boxes */
.insight-box {
    background: #1a1d2e;
    border: 1px solid #2a2d42;
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #c8c8e0;
    line-height: 1.6;
}
.insight-box.green { border-left-color: #34d399; }
.insight-box.purple { border-left-color: #a78bfa; }
.insight-box.yellow { border-left-color: #fbbf24; }
.insight-box.red { border-left-color: #f87171; }

/* Cluster summary cards */
.cluster-card {
    border-radius: 14px;
    padding: 1.2rem;
    margin: 0.5rem 0;
    border: 1px solid #2a2d42;
}

/* Plot backgrounds */
.stPlotlyChart, .stpyplot { border-radius: 12px; overflow: hidden; }

/* Selectboxes and sliders */
.stSelectbox > div, .stSlider > div {
    color: #e0e0f5;
}

/* Hide hamburger menu & footer only */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Style the top header bar to match dark theme */
[data-testid="stHeader"] {
    background: #0d0f1a !important;
    border-bottom: 1px solid #1e2035 !important;
}

/* Keep the toolbar visible but hide the deploy button */
[data-testid="stToolbar"] {
    visibility: visible !important;
}
[data-testid="stToolbar"] [data-testid="stToolbarActionButtonIcon"] {
    display: none;
}

/* ── Sidebar collapse/expand button — always visible purple pill ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    background-color: #a78bfa !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg {
    fill: white !important;
}
/* Sidebar inner collapse button */
[data-testid="stSidebarCollapseButton"] button {
    color: #a78bfa !important;
    background: transparent !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: #1e2035 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#12141f",
    plot_bgcolor="#12141f",
    font=dict(color="#c8c8e0", family="DM Sans"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor="#1e2035", zerolinecolor="#2a2d42"),
    yaxis=dict(gridcolor="#1e2035", zerolinecolor="#2a2d42"),
)

CLUSTER_COLORS = {0: "#f87171", 1: "#38bdf8", 2: "#fbbf24", 3: "#34d399"}

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color string to rgba() string for Plotly compatibility."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
CLUSTER_NAMES_DEFAULT = {0: "Budget Shoppers", 1: "High Earners", 2: "Value Seekers", 3: "Premium Buyers"}

# ─── DATA LOADING & PROCESSING ───────────────────────────────────────────────
@st.cache_data
def load_and_process(n_clusters=4):
    df = pd.read_csv("df_cleaned.csv")
    df = df[(df["Age"] < 90) & (df["Income"] < 600_000)].copy()

    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder(sparse_output=False)
    enc_cols = ohe.fit_transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_cols, columns=ohe.get_feature_names_out(cat_cols), index=df.index)
    df_encoded = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_agg = agg.fit_predict(X_pca)

    df["cluster_kmeans"] = labels_kmeans
    df["cluster_agg"] = labels_agg
    df["PCA0"] = X_pca[:, 0]
    df["PCA1"] = X_pca[:, 1]
    df["PCA2"] = X_pca[:, 2]

    # Elbow + Silhouette
    wcss, sil_scores = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        wcss.append(km.inertia_)
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, lbl))

    ev = pca.explained_variance_ratio_
    return df, X_pca, wcss, sil_scores, ev, df_encoded

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 SmartCart")
    st.markdown("**Customer Segmentation Engine**")
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Settings")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)
    algorithm = st.selectbox("Clustering Algorithm", ["K-Means", "Agglomerative"])
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🔍 Filter Data")

    df_raw = pd.read_csv("df_cleaned.csv")
    df_raw = df_raw[(df_raw["Age"] < 90) & (df_raw["Income"] < 600_000)]

    income_range = st.slider(
        "Income Range (₹)",
        int(df_raw["Income"].min()), int(df_raw["Income"].max()),
        (int(df_raw["Income"].min()), int(df_raw["Income"].max()))
    )
    education_filter = st.multiselect(
        "Education Level",
        options=df_raw["Education"].unique().tolist(),
        default=df_raw["Education"].unique().tolist()
    )
    living_filter = st.multiselect(
        "Living With",
        options=df_raw["Living_With"].unique().tolist(),
        default=df_raw["Living_With"].unique().tolist()
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏷️ Cluster Names")
    cluster_names = {}
    defaults = CLUSTER_NAMES_DEFAULT
    for i in range(n_clusters):
        cluster_names[i] = st.text_input(f"Cluster {i}", value=defaults.get(i, f"Segment {i}"))

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.caption("Built with ❤️ for SmartCart Analytics · 2025")

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
df, X_pca, wcss, sil_scores, explained_var, df_encoded = load_and_process(n_clusters)

# Apply sidebar filters
mask = (
    (df["Income"] >= income_range[0]) &
    (df["Income"] <= income_range[1]) &
    (df["Education"].isin(education_filter)) &
    (df["Living_With"].isin(living_filter))
)
df_filtered = df[mask].copy()

cluster_col = "cluster_kmeans" if algorithm == "K-Means" else "cluster_agg"
df_filtered["cluster"] = df_filtered[cluster_col]

# ─── HERO HEADER ─────────────────────────────────────────────────────────────


st.markdown('<p class="hero-title">SmartCart Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Customer Segmentation · Behaviour Intelligence · Marketing Strategy</p>', unsafe_allow_html=True)
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (f"{len(df_filtered):,}", "Total Customers"),
    (f"<span style='font-size:1.7rem'>₹{df_filtered['Income'].mean():,.0f}</span>", "Avg Income"),
    (f"₹{df_filtered['Total_Spending'].mean():,.0f}", "Avg Spending"),
    (f"{df_filtered['Age'].mean():.0f} yrs", "Avg Age"),
    (f"{n_clusters}", "Segments"),
]
for col, (val, lbl) in zip([k1, k2, k3, k4, k5], kpis):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔬 EDA",
    "📈 Optimal K",
    "🗺️ Clusters 3D",
    "👥 Segment Profiles",
    "🎯 Strategy",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Cluster Distribution
        cluster_counts = df_filtered["cluster"].value_counts().sort_index()
        colors_list = [CLUSTER_COLORS.get(i, "#888") for i in cluster_counts.index]
        fig = go.Figure(go.Bar(
            x=[cluster_names.get(i, f"Cluster {i}") for i in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=colors_list,
            text=cluster_counts.values,
            textposition="outside",
            textfont=dict(color="#e0e0f5"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Cluster Distribution", title_font_size=15)
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Income vs Spending scatter
        fig = px.scatter(
            df_filtered,
            x="Total_Spending", y="Income",
            color=df_filtered["cluster"].map(lambda x: cluster_names.get(x, f"Cluster {x}")),
            color_discrete_map={cluster_names.get(k, f"Cluster {k}"): v for k, v in CLUSTER_COLORS.items()},
            opacity=0.65,
            title="Income vs Total Spending by Segment",
        )
        fig.update_layout(**PLOTLY_LAYOUT, legend_title_text="Segment")
        st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        # Education distribution
        edu_counts = df_filtered["Education"].value_counts()
        fig = go.Figure(go.Pie(
            labels=edu_counts.index,
            values=edu_counts.values,
            hole=0.5,
            marker_colors=["#a78bfa", "#38bdf8", "#34d399"],
            textfont_color="#e0e0f5",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Education Distribution", showlegend=True)
        st.plotly_chart(fig, width='stretch')

    with col4:
        # Living With
        lw_counts = df_filtered["Living_With"].value_counts()
        fig = go.Figure(go.Pie(
            labels=lw_counts.index,
            values=lw_counts.values,
            hole=0.5,
            marker_colors=["#fbbf24", "#f87171"],
            textfont_color="#e0e0f5",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Living Situation", showlegend=True)
        st.plotly_chart(fig, width='stretch')

    # PCA Explained Variance
    st.markdown('<div class="section-title">📐 PCA Explained Variance</div>', unsafe_allow_html=True)
    ev_pct = [f"{v*100:.1f}%" for v in explained_var]
    st.markdown(f"""
    <div class="insight-box purple">
        <b>PCA Components:</b> PC1 explains <b>{explained_var[0]*100:.1f}%</b> variance · 
        PC2 explains <b>{explained_var[1]*100:.1f}%</b> · 
        PC3 explains <b>{explained_var[2]*100:.1f}%</b> · 
        <b>Total: {sum(explained_var)*100:.1f}%</b> of total variance captured in 3D.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda_tabs = st.tabs(["Distributions", "Correlation Heatmap", "Pair Relationships"])

    with eda_tabs[0]:
        num_cols = ["Income", "Age", "Total_Spending", "Recency", "NumWebPurchases",
                    "NumStorePurchases", "NumCatalogPurchases", "Customer_Tenure_Days"]
        sel_col = st.selectbox("Select Feature", num_cols)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_filtered, x=sel_col, nbins=40,
                               color_discrete_sequence=["#a78bfa"],
                               title=f"Distribution: {sel_col}")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.box(df_filtered, y=sel_col,
                         x=df_filtered["cluster"].map(lambda x: cluster_names.get(x, f"C{x}")),
                         color=df_filtered["cluster"].map(lambda x: cluster_names.get(x, f"C{x}")),
                         color_discrete_map={cluster_names.get(k, f"C{k}"): v for k, v in CLUSTER_COLORS.items()},
                         title=f"{sel_col} by Segment")
            fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Segment", showlegend=False)
            st.plotly_chart(fig, width='stretch')

    with eda_tabs[1]:
        numeric_df = df_filtered.select_dtypes(include=[np.number]).drop(
            columns=["cluster_kmeans", "cluster_agg", "cluster", "PCA0", "PCA1", "PCA2"], errors="ignore"
        )
        corr = numeric_df.corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Feature Correlation Heatmap",
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_colorbar=dict(title="r"))
        fig.update_traces(textfont_size=9)
        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        <div class="insight-box">
            <b>Key Correlations:</b> Income ↔ Total_Spending (r=0.79) · Income ↔ NumCatalogPurchases (r=0.69) · 
            NumWebVisitsMonth ↔ Income (r=-0.65, inverse) · Total_Children ↔ Total_Spending (r=-0.50)
        </div>
        """, unsafe_allow_html=True)

    with eda_tabs[2]:
        x_feat = st.selectbox("X Axis", num_cols, index=0)
        y_feat = st.selectbox("Y Axis", num_cols, index=2)
        size_feat = st.selectbox("Bubble Size", num_cols, index=3)
        fig = px.scatter(
            df_filtered,
            x=x_feat, y=y_feat,
            size=size_feat,
            color=df_filtered["cluster"].map(lambda x: cluster_names.get(x, f"C{x}")),
            color_discrete_map={cluster_names.get(k, f"C{k}"): v for k, v in CLUSTER_COLORS.items()},
            opacity=0.7,
            title=f"{x_feat} vs {y_feat}",
            size_max=20,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — OPTIMAL K
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">📈 Optimal K Selection</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=wcss,
            mode="lines+markers",
            line=dict(color="#a78bfa", width=2.5),
            marker=dict(size=8, color="#a78bfa"),
            name="WCSS",
        ))
        fig.add_vline(x=4, line_dash="dash", line_color="#fbbf24", annotation_text="Elbow → K=4")
        fig.update_layout(**PLOTLY_LAYOUT, title="Elbow Method (WCSS)", xaxis_title="K", yaxis_title="WCSS")
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(2, 11)), y=sil_scores,
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=8, color="#38bdf8"),
            name="Silhouette",
        ))
        best_k = np.argmax(sil_scores) + 2
        fig.add_vline(x=best_k, line_dash="dash", line_color="#34d399",
                      annotation_text=f"Peak → K={best_k}")
        fig.update_layout(**PLOTLY_LAYOUT, title="Silhouette Score", xaxis_title="K", yaxis_title="Score")
        st.plotly_chart(fig, width='stretch')

    # Combined chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=list(range(2, 11)), y=wcss[1:],
        mode="lines+markers",
        name="WCSS",
        line=dict(color="#a78bfa", width=2.5),
        marker=dict(size=7),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=list(range(2, 11)), y=sil_scores,
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#38bdf8", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="x"),
    ), secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title="WCSS + Silhouette Score (Combined View)")
    fig.update_yaxes(title_text="WCSS", secondary_y=False, gridcolor="#1e2035")
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, gridcolor="#1e2035")
    st.plotly_chart(fig, width='stretch')

    st.markdown(f"""
    <div class="insight-box green">
        ✅ <b>Optimal K = {n_clusters} selected.</b> The elbow point in WCSS shows the inflection at K=4 — 
        beyond this, diminishing returns on inertia reduction. The silhouette score peaks near K=5–8, 
        but K=4 balances interpretability and cluster quality (Silhouette ≈ 0.36).
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERS 3D
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<div class="section-title">🗺️ 3D PCA Cluster Visualization — {algorithm}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter_3d(
            df_filtered,
            x="PCA0", y="PCA1", z="PCA2",
            color=df_filtered["cluster"].map(lambda x: cluster_names.get(x, f"C{x}")),
            color_discrete_map={cluster_names.get(k, f"C{k}"): v for k, v in CLUSTER_COLORS.items()},
            opacity=0.7,
            title=f"3D PCA Projection — {algorithm}",
            labels={"PCA0": "PC1", "PCA1": "PC2", "PCA2": "PC3"},
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(
            paper_bgcolor="#12141f",
            plot_bgcolor="#12141f",
            font=dict(color="#c8c8e0", family="DM Sans"),
            scene=dict(
                bgcolor="#12141f",
                xaxis=dict(backgroundcolor="#12141f", gridcolor="#1e2035", color="#7c7f9e"),
                yaxis=dict(backgroundcolor="#12141f", gridcolor="#1e2035", color="#7c7f9e"),
                zaxis=dict(backgroundcolor="#12141f", gridcolor="#1e2035", color="#7c7f9e"),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=550,
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("**Cluster Sizes**")
        for cid in sorted(df_filtered["cluster"].unique()):
            cnt = (df_filtered["cluster"] == cid).sum()
            pct = cnt / len(df_filtered) * 100
            color = CLUSTER_COLORS.get(cid, "#888")
            st.markdown(f"""
            <div style="background:#1a1d2e;border:1px solid #2a2d42;border-left:4px solid {color};
                        border-radius:10px;padding:0.8rem 1rem;margin:0.4rem 0;">
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:{color}">
                    {cluster_names.get(cid, f'Cluster {cid}')}
                </div>
                <div style="font-size:1.4rem;font-weight:800;color:#e0e0f5">{cnt:,}</div>
                <div style="font-size:0.78rem;color:#7c7f9e">{pct:.1f}% of dataset</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sil = silhouette_score(
            df_filtered[["PCA0", "PCA1", "PCA2"]].values,
            df_filtered["cluster"].values
        )
        st.markdown(f"""
        <div class="insight-box purple">
            <b>Silhouette Score</b><br>
            <span style="font-size:1.8rem;font-family:'Syne',sans-serif;color:#a78bfa">{sil:.3f}</span><br>
            <span style="font-size:0.8rem;color:#7c7f9e">Higher = better separation</span>
        </div>
        """, unsafe_allow_html=True)

    # KMeans vs Agglomerative side by side
    st.markdown('<div class="section-title">⚖️ Algorithm Comparison</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        Both <b>K-Means</b> and <b>Agglomerative Clustering</b> produce visually similar 4-cluster separations in PCA space. 
        Agglomerative clustering (Ward linkage) is hierarchy-based and does not require a centroid assumption — 
        useful for non-spherical clusters. K-Means is faster and scales better for large datasets.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — SEGMENT PROFILES
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">👥 Customer Segment Deep Dive</div>', unsafe_allow_html=True)

    numeric_features = ["Income", "Age", "Total_Spending", "Recency",
                        "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases",
                        "NumDealsPurchases", "NumWebVisitsMonth", "Customer_Tenure_Days", "Total_Children"]

    summary = df_filtered.groupby("cluster")[numeric_features].mean().round(1)

    # Radar Chart
    st.markdown("#### 🕸️ Radar Chart — Multi-feature Segment Comparison")
    radar_features = ["Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases",
                      "NumCatalogPurchases", "NumDealsPurchases"]
    radar_norm = summary[radar_features].copy()
    for col in radar_features:
        radar_norm[col] = (radar_norm[col] - radar_norm[col].min()) / (radar_norm[col].max() - radar_norm[col].min() + 1e-9)

    fig = go.Figure()
    for cid in sorted(df_filtered["cluster"].unique()):
        if cid in radar_norm.index:
            vals = radar_norm.loc[cid, radar_features].tolist()
            vals += [vals[0]]
            theta = radar_features + [radar_features[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=theta,
                fill="toself",
                name=cluster_names.get(cid, f"Cluster {cid}"),
                line_color=CLUSTER_COLORS.get(cid, "#888"),
                fillcolor=hex_to_rgba(CLUSTER_COLORS.get(cid, "#888888"), alpha=0.2),
            ))
    fig.update_layout(
        polar=dict(
            bgcolor="#12141f",
            radialaxis=dict(visible=True, gridcolor="#2a2d42", color="#7c7f9e"),
            angularaxis=dict(color="#c8c8e0", gridcolor="#2a2d42"),
        ),
        paper_bgcolor="#12141f",
        font=dict(color="#c8c8e0", family="DM Sans"),
        title="Normalized Feature Radar",
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2a2d42"),
    )
    st.plotly_chart(fig, width='stretch')

    # Feature mean bars
    st.markdown("#### 📊 Feature Means by Cluster")
    feat_sel = st.selectbox("Feature to Compare", numeric_features, index=2)
    means = summary[feat_sel]
    fig = go.Figure(go.Bar(
        x=[cluster_names.get(i, f"C{i}") for i in means.index],
        y=means.values,
        marker_color=[CLUSTER_COLORS.get(i, "#888") for i in means.index],
        text=[f"{v:,.1f}" for v in means.values],
        textposition="outside",
        textfont=dict(color="#e0e0f5"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Mean {feat_sel} per Segment", yaxis_title=feat_sel)
    st.plotly_chart(fig, width='stretch')

    # Full summary table
    st.markdown("#### 📋 Full Cluster Summary Table")
    display_summary = summary.copy()
    display_summary.index = [cluster_names.get(i, f"Cluster {i}") for i in display_summary.index]
    display_summary.columns = [c.replace("_", " ") for c in display_summary.columns]
    st.dataframe(
        display_summary.style
        .background_gradient(cmap="RdPu", axis=0)
        .format("{:,.1f}"),
        width='stretch',
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — STRATEGY
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🎯 Marketing Strategy by Segment</div>', unsafe_allow_html=True)

    summary = df_filtered.groupby("cluster")[["Income", "Total_Spending", "Age",
                                               "NumWebPurchases", "NumStorePurchases",
                                               "NumCatalogPurchases", "NumDealsPurchases",
                                               "NumWebVisitsMonth", "Total_Children"]].mean().round(1)

    strategy_colors = [CLUSTER_COLORS.get(i, "#888") for i in range(n_clusters)]
    color_labels = ["red", "blue", "yellow", "green"]

    for cid in sorted(df_filtered["cluster"].unique()):
        if cid not in summary.index:
            continue
        row = summary.loc[cid]
        name = cluster_names.get(cid, f"Cluster {cid}")
        color = CLUSTER_COLORS.get(cid, "#888")
        cnt = int((df_filtered["cluster"] == cid).sum())

        # Build automatic strategy insights
        strategy_lines = []
        if row["Income"] > summary["Income"].mean():
            strategy_lines.append("💎 High income — target with <b>premium & luxury</b> product lines.")
        else:
            strategy_lines.append("💰 Budget-conscious — emphasize <b>deals, discounts & value bundles</b>.")

        if row["NumWebPurchases"] > summary["NumWebPurchases"].mean():
            strategy_lines.append("🌐 Web-savvy — invest in <b>email campaigns, personalized web ads</b>.")
        else:
            strategy_lines.append("🏪 Store-loyal — strengthen <b>in-store promotions & loyalty cards</b>.")

        if row["Total_Children"] > 0.5:
            strategy_lines.append("👶 Family segment — promote <b>family packs, back-to-school offers</b>.")
        else:
            strategy_lines.append("👤 Child-free — focus on <b>lifestyle & self-indulgence products</b>.")

        if row["NumDealsPurchases"] > summary["NumDealsPurchases"].mean():
            strategy_lines.append("🏷️ Deal-seekers — <b>flash sales, coupons & loyalty discounts</b> work well.")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1d2e,#161827);
                    border:1px solid #2a2d42;border-top:4px solid {color};
                    border-radius:14px;padding:1.4rem 1.6rem;margin:0.8rem 0;">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:{color}">
                    {name}
                </div>
                <div style="font-size:0.78rem;color:#7c7f9e;border:1px solid #2a2d42;
                            border-radius:20px;padding:0.2rem 0.7rem;">
                    {cnt:,} customers
                </div>
                <div style="font-size:0.78rem;color:#7c7f9e;border:1px solid #2a2d42;
                            border-radius:20px;padding:0.2rem 0.7rem;">
                    Avg Income: ₹{row['Income']:,.0f}
                </div>
                <div style="font-size:0.78rem;color:#7c7f9e;border:1px solid #2a2d42;
                            border-radius:20px;padding:0.2rem 0.7rem;">
                    Avg Spend: ₹{row['Total_Spending']:,.0f}
                </div>
            </div>
            <ul style="margin:0;padding-left:1.2rem;color:#c8c8e0;font-size:0.92rem;line-height:2;">
                {''.join(f'<li>{s}</li>' for s in strategy_lines)}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Overall Recommendations</div>', unsafe_allow_html=True)

    recs = [
        ("🎯 Personalization", "Use cluster labels to personalize homepage product recommendations — customers in high-income clusters should see premium products first.", "purple"),
        ("📧 Email Segmentation", "Split your mailing list by cluster. High-frequency web visitors (Cluster 1/3) respond best to retargeting emails with product reminders.", "blue"),
        ("🏷️ Pricing Strategy", "Budget clusters benefit from bundled pricing and BOGO offers. Premium clusters respond to exclusive access and early sale notifications.", "yellow"),
        ("📱 Channel Mix", "Web-purchase dominant clusters → push notifications + display ads. Store-purchase clusters → in-store loyalty programs + paper catalogues.", "green"),
    ]
    for title, body, cls in recs:
        st.markdown(f'<div class="insight-box {cls}"><b>{title}:</b> {body}</div>', unsafe_allow_html=True)

    # Download section
    st.markdown('<div class="section-title">⬇️ Export Data</div>', unsafe_allow_html=True)
    export_df = df_filtered[["Income", "Age", "Total_Spending", "Recency", "Education",
                              "Living_With", "cluster"]].copy()
    export_df["Segment_Name"] = export_df["cluster"].map(cluster_names)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Segmented Customer Data (CSV)",
        data=csv,
        file_name="smartcart_segmented_customers.csv",
        mime="text/csv",
    )

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#3d4060;font-size:0.8rem;padding:0.5rem 0;">
    SmartCart Customer Segmentation · Built with Streamlit · K-Means & Agglomerative Clustering · PCA Dimensionality Reduction
</div>
""", unsafe_allow_html=True)