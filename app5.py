
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Hyundai Distribution Analytics", page_icon="🚗", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Hyundai Distribution Analytics")
st.markdown("**Real-time Lead Time Analysis & Outlier Detection**")
st.markdown("---")

# ==================== HELPER FUNCTIONS ====================
def _first_col(df, candidates):
    cols = list(df.columns)
    for cand in candidates:
        for col in cols:
            if cand.lower() in col.lower():
                return col
    return None

def _parse_date(df, col):
    if col is None or col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    dt = pd.to_datetime(df[col], errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return dt

def mad_sigma(x):
    x = pd.to_numeric(x, errors='coerce').dropna()
    if x.empty:
        return 0.0
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))

def classify_region_zone(region):
    if pd.isna(region):
        return "Unknown"
    region_lower = str(region).lower()
    if any(x in region_lower for x in ['sumatra', 'sumatera', 'jawa', 'java', 'bali']):
        return "Indonesia Barat"
    elif any(x in region_lower for x in ['kalimantan', 'sulawesi', 'nusa tenggara']):
        return "Indonesia Tengah"
    elif any(x in region_lower for x in ['papua', 'maluku']):
        return "Indonesia Timur"
    return "Unknown"

# ==================== MAPPING PRIORITY ====================
PRIORITY = {
    "id_col": ["VIN", "No", "shipment_id", "id"],
    "model_col": ["Model", "product", "vehicle"],
    "transporter_col": ["Transporter", "carrier", "transport"],
    "tos_col": ["TOS", "mode"],
    "tos_date_col": ["TOS Date", "Date TOS", "Target Gate Out", "Gate Out", "start"],
    "ata_date_col": ["Date ATA", "ATA", "delivery", "actual_arrival", "actual arrival"],
    "start_date_col": ["Date ETD", "ETD", "Target Gate Out", "start"],
    "end_date_col": ["Date ATA", "ATA", "delivery", "actual_arrival"],
    "dest_col": ["Outlet/PDC", "Outlet", "PDC", "Destination", "dealer"],
    "city_col": ["City", "Kota", "destination_city"],
    "province_col": ["Province", "Provinsi"],
    "region_col": ["Region", "Island", "area"]
}

def build_mapping(df):
    mapping = {}
    for key, cands in PRIORITY.items():
        mapping[key] = _first_col(df, cands)
    return mapping, []

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        st.success(f"✅ {len(df_raw):,} rows loaded")
        
        semantic_map, _ = build_mapping(df_raw)
        
        st.markdown("---")
        st.subheader("🔍 Column Mapping")
        
        nice_names = {
            "id_col": "🚗 VIN/ID",
            "model_col": "📦 Model",
            "city_col": "🏙️ City",
            "tos_date_col": "📅 Start Date",
            "ata_date_col": "📅 End Date",
            "dest_col": "📍 Destination",
            "transporter_col": "🚚 Transporter",
            "region_col": "🌏 Region"
        }
        
        def select_col(label, key):
            opts = ["— None —"] + list(df_raw.columns)
            default = semantic_map.get(key)
            idx = opts.index(default) if default in opts else 0
            return st.selectbox(label, opts, index=idx, key=f"map_{key}") if opts[idx] != "— None —" else None
        
        for k in ["id_col", "model_col", "city_col", "tos_date_col", "ata_date_col", "dest_col"]:
            semantic_map[k] = select_col(nice_names.get(k, k), k)
        
        with st.expander("Optional Columns"):
            for k in ["transporter_col", "region_col"]:
                semantic_map[k] = select_col(nice_names.get(k, k), k)
        
        start_key = semantic_map.get("tos_date_col") or semantic_map.get("start_date_col")
        end_key = semantic_map.get("ata_date_col") or semantic_map.get("end_date_col")
        
        required_ok = all([semantic_map.get("id_col"), semantic_map.get("model_col"), 
                          semantic_map.get("dest_col"), start_key, end_key])
        
        st.markdown("---")
        st.subheader("⚙️ Detection Method")
        
        detection_method = st.radio(
            "Method",
            ["Sigma-based", "SLA-based", "Combined"],
            index=2
        )
        
        if detection_method in ["Sigma-based", "Combined"]:
            mad_threshold = st.slider("Sigma (σ)", 1.0, 5.0, 3.0, 0.5)
        else:
            mad_threshold = None
        
        if detection_method in ["SLA-based", "Combined"]:
            st.markdown("**SLA Standards (days):**")
            sla_barat = st.number_input("Barat", 1, 30, 5)
            sla_tengah = st.number_input("Tengah", 1, 30, 10)
            sla_timur = st.number_input("Timur", 1, 30, 15)
            sla_config = {"Indonesia Barat": sla_barat, "Indonesia Tengah": sla_tengah, "Indonesia Timur": sla_timur}
        else:
            sla_config = None
        
        analyze_btn = st.button("🚀 Analyze Data", type="primary", use_container_width=True)

# ==================== MAIN CONTENT ====================
if uploaded_file is not None and analyze_btn and required_ok:
    with st.spinner("🔄 Processing..."):
        df = df_raw.copy()
        
        start_dt = _parse_date(df, start_key)
        end_dt = _parse_date(df, end_key)
        
        df["start_date_parsed"] = start_dt
        df["end_date_parsed"] = end_dt
        df["lead_time_calc"] = (end_dt - start_dt).dt.days
        
        df = df[(df["lead_time_calc"].notna()) & (df["lead_time_calc"] >= 0)].copy()
        
        if not df.empty:
            df["year"] = df["start_date_parsed"].dt.year
            df["month"] = df["start_date_parsed"].dt.month
            df["month_start"] = df["start_date_parsed"].dt.to_period("M").dt.to_timestamp()
            df["month_name"] = df["month_start"].dt.strftime("%b %Y")
            df["quarter"] = df["start_date_parsed"].dt.quarter
            df["day_of_week"] = df["start_date_parsed"].dt.day_name()
            
            if semantic_map.get("region_col"):
                df["zone"] = df[semantic_map["region_col"]].apply(classify_region_zone)
            else:
                df["zone"] = "Unknown"
            
            # Outlier detection
            if detection_method == "SLA-based":
                df["sla_standard"] = df["zone"].map(sla_config).fillna(10)
                df["is_outlier"] = df["lead_time_calc"] > df["sla_standard"]
                df["outlier_method"] = "SLA"
            elif detection_method == "Sigma-based":
                med = df["lead_time_calc"].median()
                mad = mad_sigma(df["lead_time_calc"])
                if mad > 0:
                    df["is_outlier"] = np.abs(df["lead_time_calc"] - med) > (mad_threshold * mad)
                else:
                    q1 = df["lead_time_calc"].quantile(0.25)
                    q3 = df["lead_time_calc"].quantile(0.75)
                    iqr = q3 - q1
                    df["is_outlier"] = (df["lead_time_calc"] < q1 - 1.5*iqr) | (df["lead_time_calc"] > q3 + 1.5*iqr)
                df["outlier_method"] = "Sigma"
            else:  # Combined
                df["sla_standard"] = df["zone"].map(sla_config).fillna(10)
                df["is_outlier_sla"] = df["lead_time_calc"] > df["sla_standard"]
                med = df["lead_time_calc"].median()
                mad = mad_sigma(df["lead_time_calc"])
                if mad > 0:
                    df["is_outlier_sigma"] = np.abs(df["lead_time_calc"] - med) > (mad_threshold * mad)
                else:
                    q1 = df["lead_time_calc"].quantile(0.25)
                    q3 = df["lead_time_calc"].quantile(0.75)
                    iqr = q3 - q1
                    df["is_outlier_sigma"] = (df["lead_time_calc"] < q1 - 1.5*iqr) | (df["lead_time_calc"] > q3 + 1.5*iqr)
                df["is_outlier"] = df["is_outlier_sla"] | df["is_outlier_sigma"]
                df["outlier_method"] = "None"
                df.loc[df["is_outlier_sla"] & df["is_outlier_sigma"], "outlier_method"] = "Both"
                df.loc[df["is_outlier_sla"] & ~df["is_outlier_sigma"], "outlier_method"] = "SLA Only"
                df.loc[~df["is_outlier_sla"] & df["is_outlier_sigma"], "outlier_method"] = "Sigma Only"
            
            st.session_state["df_analyzed"] = df
            st.session_state["config"] = semantic_map
            st.session_state["detection_method"] = detection_method

if "df_analyzed" in st.session_state:
    df = st.session_state["df_analyzed"]
    config = st.session_state["config"]
    detection_method = st.session_state.get("detection_method", "Combined")
    
    # Method indicator
    method_colors = {"SLA-based": "#dc3545", "Sigma-based": "#667eea", "Combined": "#28a745"}
    st.markdown(f"""
    <div style='background:{method_colors.get(detection_method, "#667eea")};color:white;padding:10px;border-radius:5px;text-align:center;'>
        <strong>Detection Method:</strong> {detection_method}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Performance", "🚨 Outlier Analysis", "🏙️ City Deep Dive"])
    
    # ===== TAB 1: DASHBOARD =====
    with tab1:
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        total = len(df)
        avg_lt = df["lead_time_calc"].mean()
        median_lt = df["lead_time_calc"].median()
        outliers = int(df["is_outlier"].sum())
        outlier_pct = (outliers/total*100) if total > 0 else 0
        
        with col1:
            st.metric("📦 Total Shipments", f"{total:,}")
        with col2:
            st.metric("⏱️ Avg Lead Time", f"{avg_lt:.1f}d")
        with col3:
            st.metric("📊 Median LT", f"{median_lt:.1f}d")
        with col4:
            st.metric("🚨 Outliers", f"{outliers:,}")
        with col5:
            st.metric("📈 Outlier Rate", f"{outlier_pct:.1f}%")
        
        st.markdown("---")
        
        # Lead Time Trend (full width)
        st.subheader("📈 Lead Time Trend")
        monthly = df.groupby(["month_start", "month_name"])["lead_time_calc"].agg(['mean', 'median', 'count']).reset_index()
        monthly = monthly.sort_values("month_start")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["count"], name="Volume", 
                            marker_color='rgba(100, 181, 246, 0.6)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["mean"], name="Avg Lead Time",
                                mode="lines+markers", line=dict(color='rgb(255, 87, 34)', width=4),
                                marker=dict(size=10, color='rgb(255, 87, 34)')), secondary_y=True)
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["median"], name="Median Lead Time",
                                mode="lines+markers", line=dict(color='rgb(76, 175, 80)', width=3, dash='dash'),
                                marker=dict(size=8, color='rgb(76, 175, 80)')), secondary_y=True)
        fig.update_layout(
            height=400, 
            hovermode="x unified", 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        fig.update_yaxes(title_text="Lead Time (days)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top Cities (full width)
        st.subheader("🏙️ Top 10 Cities by Volume")
        if config.get("city_col"):
            top_city = df.groupby(config["city_col"]).agg({
                "lead_time_calc": "mean",
                config["id_col"]: "count"
            }).reset_index()
            top_city.columns = ["City", "Avg LT", "Volume"]
            top_city = top_city.sort_values("Volume", ascending=False).head(10)
            fig = px.bar(top_city, y="City", x="Volume", orientation="h", text="Volume",
                        color="Avg LT", color_continuous_scale="RdYlGn_r")
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400, showlegend=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top Models (full width)
        st.subheader("🚗 Top 10 Models by Volume")
        top_model = df.groupby(config["model_col"]).agg({
            "lead_time_calc": "mean",
            config["id_col"]: "count"
        }).reset_index()
        top_model.columns = ["Model", "Avg LT", "Volume"]
        top_model = top_model.sort_values("Volume", ascending=False).head(10)
        fig = px.bar(top_model, y="Model", x="Volume", orientation="h", text="Volume",
                    color="Avg LT", color_continuous_scale="RdYlGn_r")
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: PERFORMANCE =====
    with tab2:
        st.header("🎯 Performance Analysis")
        
        group_options = []
        group_map = {}
        
        if config.get("city_col"):
            group_options.append("City")
            group_map["City"] = config["city_col"]
        
        group_options.append("Model")
        group_map["Model"] = config["model_col"]
        
        if config.get("transporter_col"):
            group_options.append("Transporter")
            group_map["Transporter"] = config["transporter_col"]
        
        selected_group = st.selectbox("📊 Group By", options=group_options)
        group_col = group_map[selected_group]
        
        perf_data = df.groupby(group_col).agg({
            "lead_time_calc": ["mean", "median", "std"],
            config["id_col"]: "count",
            "is_outlier": "sum"
        }).reset_index()
        perf_data.columns = [selected_group, "Avg LT", "Median LT", "Std Dev", "Volume", "Outliers"]
        perf_data["Outlier %"] = (perf_data["Outliers"] / perf_data["Volume"] * 100).round(1)
        perf_data = perf_data.sort_values("Avg LT", ascending=False).head(20)
        
        # Chart
        fig = px.bar(perf_data, y=selected_group, x="Avg LT", orientation="h", text="Avg LT",
                    color="Avg LT", color_continuous_scale="RdYlGn_r", hover_data=["Volume", "Outlier %"])
        fig.update_traces(texttemplate='%{text:.1f}d', textposition='outside')
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"📊 Detailed Statistics")
        st.dataframe(
            perf_data.style.background_gradient(cmap="RdYlGn_r", subset=["Avg LT"])
                           .background_gradient(cmap="Reds", subset=["Outlier %"])
                           .format({"Avg LT": "{:.1f}", "Median LT": "{:.1f}", "Std Dev": "{:.1f}", "Outlier %": "{:.1f}%"}),
            use_container_width=True, hide_index=True, height=400
        )
    
    # ===== TAB 3: OUTLIER ANALYSIS =====
    with tab3:
        st.header("🚨 Outlier Analysis")
        
        outliers_df = df[df["is_outlier"]].copy()
        
        if len(outliers_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Outliers", f"{len(outliers_df):,}")
            with col2:
                st.metric("Avg Lead Time", f"{outliers_df['lead_time_calc'].mean():.1f}d")
            with col3:
                st.metric("Max Lead Time", f"{outliers_df['lead_time_calc'].max():.0f}d")
            with col4:
                if detection_method == "Combined" and "outlier_method" in outliers_df.columns:
                    both_count = (outliers_df["outlier_method"] == "Both").sum()
                    st.metric("Both Methods", f"{both_count}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_filter = st.multiselect("🚗 Filter by Model",
                    options=sorted(outliers_df[config["model_col"]].dropna().unique()),
                    default=list(outliers_df[config["model_col"]].dropna().unique())
                )
            
            with col2:
                if config.get("city_col"):
                    city_filter = st.multiselect("🏙️ Filter by City",
                        options=sorted(outliers_df[config["city_col"]].dropna().unique()),
                        default=list(outliers_df[config["city_col"]].dropna().unique())
                    )
                else:
                    city_filter = None
            
            outliers_filtered = outliers_df[outliers_df[config["model_col"]].isin(model_filter)].copy()
            if city_filter and config.get("city_col"):
                outliers_filtered = outliers_filtered[outliers_filtered[config["city_col"]].isin(city_filter)]
            
            st.info(f"📊 Showing {len(outliers_filtered):,} of {len(outliers_df):,} outliers")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚗 Top 10 Models")
                top_model_outlier = outliers_filtered[config["model_col"]].value_counts().head(10)
                fig = px.bar(x=top_model_outlier.values, y=top_model_outlier.index, orientation="h",
                            text=top_model_outlier.values, color_discrete_sequence=['rgb(244, 67, 54)'])
                fig.update_traces(textposition='outside', marker=dict(line=dict(width=0)))
                fig.update_layout(height=350, showlegend=False, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if config.get("city_col"):
                    st.subheader("🏙️ Top 10 Cities")
                    top_city_outlier = outliers_filtered[config["city_col"]].value_counts().head(10)
                    fig = px.bar(x=top_city_outlier.values, y=top_city_outlier.index, orientation="h",
                                text=top_city_outlier.values, color_discrete_sequence=['rgb(255, 152, 0)'])
                    fig.update_traces(textposition='outside', marker=dict(line=dict(width=0)))
                    fig.update_layout(height=350, showlegend=False, yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Complete Outlier Records")
            outliers_display = outliers_filtered.sort_values("lead_time_calc", ascending=False)
            st.dataframe(outliers_display, use_container_width=True, height=400)
            
            csv = outliers_display.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Outliers", csv, 
                             f"outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        else:
            st.success("🎉 No outliers detected!")
    
    # ===== TAB 4: CITY DEEP DIVE =====
    with tab4:
        st.header("🏙️ City-Level Deep Dive Analysis")
        
        if config.get("city_col"):
            # City selector
            all_cities = sorted(df[config["city_col"]].dropna().unique())
            selected_city = st.selectbox("🔍 Select City for Detailed Analysis", options=all_cities)
            
            if selected_city:
                city_df = df[df[config["city_col"]] == selected_city].copy()
                city_outliers = city_df[city_df["is_outlier"]]
                
                st.markdown(f"### 📍 {selected_city}")
                
                # City KPIs
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Shipments", f"{len(city_df):,}")
                with col2:
                    st.metric("Avg Lead Time", f"{city_df['lead_time_calc'].mean():.1f}d")
                with col3:
                    st.metric("Median LT", f"{city_df['lead_time_calc'].median():.1f}d")
                with col4:
                    st.metric("Outliers", f"{len(city_outliers):,}")
                with col5:
                    city_outlier_pct = (len(city_outliers)/len(city_df)*100) if len(city_df) > 0 else 0
                    st.metric("Outlier Rate", f"{city_outlier_pct:.1f}%")
                
                st.markdown("---")
                
                # City visualizations (vertical stack)
                st.subheader("📊 Lead Time Distribution")
                fig = px.histogram(city_df, x="lead_time_calc", nbins=30, 
                                  color="is_outlier", color_discrete_map={True: 'rgb(244, 67, 54)', False: 'rgb(76, 175, 80)'},
                                  labels={"is_outlier": "Outlier"})
                fig.update_layout(height=350, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.subheader("🚗 Performance by Model")
                model_perf = city_df.groupby(config["model_col"]).agg({
                    "lead_time_calc": "mean",
                    config["id_col"]: "count"
                }).reset_index()
                model_perf.columns = ["Model", "Avg LT", "Volume"]
                model_perf = model_perf.sort_values("Avg LT", ascending=False).head(10)
                
                fig = px.bar(model_perf, y="Model", x="Avg LT", orientation="h", text="Avg LT",
                            color="Avg LT", color_continuous_scale="RdYlGn_r")
                fig.update_traces(texttemplate='%{text:.1f}d', textposition='outside')
                fig.update_layout(height=350, showlegend=False, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                if len(city_outliers) > 0:
                    st.subheader(f"🚨 Outlier Details for {selected_city}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Models with Outliers**")
                        model_outlier_count = city_outliers[config["model_col"]].value_counts().head(5)
                        for model, count in model_outlier_count.items():
                            st.write(f"• {model}: **{count}** outliers")
                    
                    with col2:
                        if config.get("transporter_col"):
                            st.markdown("**Top Transporters with Outliers**")
                            trans_outlier_count = city_outliers[config["transporter_col"]].value_counts().head(5)
                            for trans, count in trans_outlier_count.items():
                                st.write(f"• {trans}: **{count}** outliers")
                    
                    st.markdown("---")
                    st.markdown("**Complete Outlier Records**")
                    
                    # Remove zone column from display
                    display_cols = [col for col in city_outliers.columns if col != 'zone']
                    city_outliers_display = city_outliers[display_cols].sort_values("lead_time_calc", ascending=False)
                    
                    st.dataframe(city_outliers_display, use_container_width=True, height=300)
                    
                    csv_city = city_outliers.to_csv(index=False).encode("utf-8")
                    st.download_button(f"📥 Download {selected_city} Outliers", csv_city,
                                     f"outliers_{selected_city}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
                else:
                    st.success(f"✅ No outliers in {selected_city}!")
        else:
            st.warning("⚠️ City column not mapped. Please map City column in sidebar to use this feature.")

else:
    # Welcome screen
    st.info("👆 Upload your data file to begin analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Required Columns
        
        **Mandatory:**
        - **VIN/ID** - Unique vehicle identifier
        - **TOS Date** or **Date ETD** - Start date
        - **ATA Date** or **Date ATA** - End/delivery date
        - **Model** - Vehicle model
        - **Destination** - Outlet/PDC/Dealer
        
        **Highly Recommended:**
        - **City** - For city-level analysis
        - **Region** - For zone classification
        - **Transporter** - For carrier analysis
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Dashboard Features
        
        **📊 Dashboard Tab:**
        - Real-time KPIs & metrics
        - Monthly trend analysis
        - Top performers by city & model
        - Zone-based performance
        
        **🎯 Performance Tab:**
        - Dynamic grouping analysis
        - Detailed statistics table
        - Outlier rate tracking
        
        **🚨 Outlier Analysis Tab:**
        - Advanced filtering
        - Model & city breakdown
        - Downloadable reports
        
        **🏙️ City Deep Dive Tab:**
        - City-specific analysis
        - Lead time distribution
        - Model performance per city
        - City-level outlier details
        """)
    
    st.markdown("---")
    st.subheader("📄 Example Data Format")
    
    example = pd.DataFrame({
        'VIN': ['KMHXX00XXXX000001', 'KMHXX00XXXX000002', 'KMHXX00XXXX000003'],
        'Model': ['CRETA', 'STARGAZER', 'IONIQ 5'],
        'TOS Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'ATA Date': ['2024-01-05', '2024-01-12', '2024-01-15'],
        'City': ['Jakarta', 'Surabaya', 'Medan'],
        'Region': ['Jawa', 'Jawa', 'Sumatra'],
        'Destination': ['Dealer A', 'Dealer B', 'Dealer C'],
        'Transporter': ['Transporter X', 'Transporter Y', 'Transporter Z']
    })
    
    st.dataframe(example, use_container_width=True)
    
    csv_example = example.to_csv(index=False).encode('utf-8')
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            "📥 Download Example Template",
            csv_example,
            'hyundai_distribution_template.csv',
            'text/csv',
            use_container_width=True
        )
