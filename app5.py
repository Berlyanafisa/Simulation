import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Hyundai Distribution Analytics", page_icon="🚗", layout="wide")

st.title("🚗 Hyundai Distribution Analytics Dashboard")
st.markdown("**Real-time Analysis with Standardization Simulation**")
st.markdown("---")

# ==================== HELPER FUNCTIONS ====================
def _first_col(df, candidates):
    """Ambil kolom pertama yang cocok dari list kandidat"""
    cols = list(df.columns)
    for cand in candidates:
        for col in cols:
            if cand.lower() in col.lower():
                return col
    return None

def _parse_date(df, col):
    """Parse tanggal dengan berbagai format"""
    if col is None or col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    dt = pd.to_datetime(df[col], errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return dt

def mad_sigma(x):
    """MAD sigma untuk outlier detection"""
    x = pd.to_numeric(x, errors='coerce').dropna()
    if x.empty:
        return 0.0
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))

def classify_region_zone(region):
    """Klasifikasi region ke zona (Barat/Tengah/Timur)"""
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

    # NEW preferred date columns
    "tos_date_col": ["TOS Date", "Date TOS", "Target Gate Out", "Gate Out", "start"],
    "ata_date_col": ["Date ATA", "ATA", "delivery", "actual_arrival", "actual arrival"],

    # Legacy (fallback) date columns for backward-compat
    "start_date_col": ["Date ETD", "ETD", "Target Gate Out", "start"],
    "actual_depart_col": ["Date ATD", "ATD", "actual_departure"],
    "eta_col": ["Date ETA", "ETA", "estimated_arrival"],
    "end_date_col": ["Date ATA", "ATA", "delivery", "actual_arrival"],

    "dest_col": ["Outlet/PDC", "Outlet", "PDC", "Destination", "dealer"],
    "city_col": ["City", "Kota", "destination_city"],
    "province_col": ["Province", "Provinsi"],
    "region_col": ["Region", "Island", "area"]
}

def build_mapping(df):
    """Build column mapping"""
    mapping = {}
    for key, cands in PRIORITY.items():
        mapping[key] = _first_col(df, cands)
    # Required akan dicek fleksibel nanti (TOS/ETD dan ATA/ATA legacy)
    return mapping, []  # missing dihitung ulang setelah user confirm

# ==================== SIDEBAR: UPLOAD ====================
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded: {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        
        # Auto-map
        semantic_map, _ = build_mapping(df_raw)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Column Mapping")
        
        # Confirm mapping
        nice_names = {
            "id_col": "🚗 VIN/ID",
            "model_col": "📦 Model",
            "transporter_col": "🚚 Transporter",
            "tos_col": "🛣️ TOS/Mode",

            "tos_date_col": "📅 TOS Date (Start • preferred)",
            "ata_date_col": "📅 ATA Date (End • preferred)",

            "start_date_col": "📅 Date ETD (legacy/optional)",
            "actual_depart_col": "📅 Date ATD (optional)",
            "eta_col": "📅 Date ETA (optional)",
            "end_date_col": "📅 Date ATA (legacy/optional)",

            "dest_col": "📍 Outlet/PDC",
            "city_col": "🏙️ City",
            "province_col": "🗺️ Province",
            "region_col": "🌏 Region"
        }
        
        def select_col(label, key):
            opts = ["— None —"] + list(df_raw.columns)
            default = semantic_map.get(key)
            idx = opts.index(default) if default in opts else 0
            chosen = st.sidebar.selectbox(label, opts, index=idx, key=f"map_{key}")
            return None if chosen == "— None —" else chosen
        
        chosen = {k: select_col(label, k) for k, label in nice_names.items()}
        semantic_map.update(chosen)

        # ===== Flexible required check =====
        # Need ID, DEST, MODEL, and START/END where:
        # start_key = TOS Date (preferred) or ETD (legacy)
        # end_key   = ATA Date (preferred) or ATA (legacy)
        start_key = semantic_map.get("tos_date_col") or semantic_map.get("start_date_col")
        end_key   = semantic_map.get("ata_date_col") or semantic_map.get("end_date_col")

        missing_labels = []
        if semantic_map.get("id_col") is None:
            missing_labels.append(nice_names["id_col"])
        if semantic_map.get("model_col") is None:
            missing_labels.append(nice_names["model_col"])
        if semantic_map.get("dest_col") is None:
            missing_labels.append(nice_names["dest_col"])
        if start_key is None:
            missing_labels.append("📅 TOS Date (Start) **or** 📅 Date ETD")
        if end_key is None:
            missing_labels.append("📅 ATA Date (End) **or** 📅 Date ATA")

        if missing_labels:
            st.sidebar.warning("⚠️ Required: " + ", ".join(missing_labels))
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Outlier Detection Method")
        
        detection_method = st.sidebar.radio(
            "Choose Method",
            ["Sigma-based (Statistical)", "SLA-based (Custom Standards)", "Both (Combined)"],
            index=2,
            help="Select how to detect outliers"
        )
        
        if detection_method in ["Sigma-based (Statistical)", "Both (Combined)"]:
            mad_threshold = st.sidebar.slider("Sigma Threshold (σ)", 1.0, 5.0, 3.0, 0.5)
        else:
            mad_threshold = None
        
        if detection_method in ["SLA-based (Custom Standards)", "Both (Combined)"]:
            st.sidebar.markdown("**Set SLA by Zone (days):**")
            sla_barat = st.sidebar.number_input("Indonesia Barat", 1, 30, 5, key="sla_barat")
            sla_tengah = st.sidebar.number_input("Indonesia Tengah", 1, 30, 10, key="sla_tengah")
            sla_timur = st.sidebar.number_input("Indonesia Timur", 1, 30, 15, key="sla_timur")
            sla_config = {"Indonesia Barat": sla_barat, "Indonesia Tengah": sla_tengah, "Indonesia Timur": sla_timur}
        else:
            sla_config = None
        
        # ==================== ANALYZE BUTTON ====================
        if st.sidebar.button("🚀 Analyze Data", type="primary"):
            if missing_labels:
                st.error("❌ Please complete required columns in sidebar")
            else:
                with st.spinner("Processing data..."):
                    df = df_raw.copy()
                    
                    # Parse dates (TOS Date preferred, fallback ETD; ATA Date preferred, fallback ATA)
                    start_dt = _parse_date(df, start_key)
                    end_dt   = _parse_date(df, end_key)
                    
                    df["start_date_parsed"] = start_dt
                    df["end_date_parsed"] = end_dt
                    df["lead_time_calc"] = (end_dt - start_dt).dt.days
                    
                    # Clean
                    df = df[(df["lead_time_calc"].notna()) & (df["lead_time_calc"] >= 0)].copy()
                    
                    if df.empty:
                        st.error("❌ No valid data after cleaning")
                    else:
                        # Time features
                        df["year"] = df["start_date_parsed"].dt.year
                        df["month"] = df["start_date_parsed"].dt.month
                        df["month_start"] = df["start_date_parsed"].dt.to_period("M").dt.to_timestamp()
                        df["month_name"] = df["month_start"].dt.strftime("%b %Y")
                        df["quarter"] = df["start_date_parsed"].dt.quarter
                        df["day_of_week"] = df["start_date_parsed"].dt.day_name()
                        
                        # Classify zone
                        if semantic_map.get("region_col"):
                            df["zone"] = df[semantic_map["region_col"]].apply(classify_region_zone)
                        else:
                            df["zone"] = "Unknown"
                        
                        # Outlier detection based on method
                        if detection_method == "SLA-based (Custom Standards)":
                            # SLA only
                            df["sla_standard"] = df["zone"].map(sla_config).fillna(10)
                            df["is_outlier"] = df["lead_time_calc"] > df["sla_standard"]
                            df["outlier_method"] = "SLA"
                            
                        elif detection_method == "Sigma-based (Statistical)":
                            # Sigma only
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
                            
                        else:  # Both (Combined)
                            # SLA based
                            df["sla_standard"] = df["zone"].map(sla_config).fillna(10)
                            df["is_outlier_sla"] = df["lead_time_calc"] > df["sla_standard"]
                            
                            # Sigma based
                            med = df["lead_time_calc"].median()
                            mad = mad_sigma(df["lead_time_calc"])
                            
                            if mad > 0:
                                df["is_outlier_sigma"] = np.abs(df["lead_time_calc"] - med) > (mad_threshold * mad)
                            else:
                                q1 = df["lead_time_calc"].quantile(0.25)
                                q3 = df["lead_time_calc"].quantile(0.75)
                                iqr = q3 - q1
                                df["is_outlier_sigma"] = (df["lead_time_calc"] < q1 - 1.5*iqr) | (df["lead_time_calc"] > q3 + 1.5*iqr)
                            
                            # Combined
                            df["is_outlier"] = df["is_outlier_sla"] | df["is_outlier_sigma"]
                            
                            # Tag method
                            df["outlier_method"] = "None"
                            df.loc[df["is_outlier_sla"] & df["is_outlier_sigma"], "outlier_method"] = "Both"
                            df.loc[df["is_outlier_sla"] & ~df["is_outlier_sigma"], "outlier_method"] = "SLA Only"
                            df.loc[~df["is_outlier_sla"] & df["is_outlier_sigma"], "outlier_method"] = "Sigma Only"
                        
                        # Store
                        st.session_state["df_analyzed"] = df
                        st.session_state["config"] = semantic_map
                        st.session_state["detection_method"] = detection_method
                        
                        # Success message with method info
                        if detection_method == "SLA-based (Custom Standards)":
                            st.success(f"✅ Analysis complete! {len(df):,} records processed using SLA standards")
                        elif detection_method == "Sigma-based (Statistical)":
                            st.success(f"✅ Analysis complete! {len(df):,} records processed using {mad_threshold}σ threshold")
                        else:
                            st.success(f"✅ Analysis complete! {len(df):,} records processed using combined method")
        
        # ==================== DISPLAY TABS ====================
        if "df_analyzed" in st.session_state:
            df = st.session_state["df_analyzed"]
            config = st.session_state["config"]
            detection_method = st.session_state.get("detection_method", "Both (Combined)")
            
            # Show detection method info
            if detection_method == "SLA-based (Custom Standards)":
                st.info("🔍 **Detection Method**: SLA-based only (Sigma rules disabled)")
            elif detection_method == "Sigma-based (Statistical)":
                st.info("🔍 **Detection Method**: Sigma-based only (SLA standards disabled)")
            else:
                st.info("🔍 **Detection Method**: Combined (SLA + Sigma)")
            
            # REMOVED: SLA Simulation tab
            tab1, tab2, tab3 = st.tabs([
                "📊 Overview",
                "🎯 Performance Analysis",
                "🚨 Outlier Analysis"
            ])
            
            # ===== TAB 1: OVERVIEW =====
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                total = len(df)
                avg_lt = df["lead_time_calc"].mean()
                median_lt = df["lead_time_calc"].median()
                outliers = int(df["is_outlier"].sum())
                
                with col1:
                    st.metric("Total Shipments", f"{total:,}")
                with col2:
                    st.metric("Avg Lead Time", f"{avg_lt:.1f} days")
                with col3:
                    st.metric("Median Lead Time", f"{median_lt:.1f} days")
                with col4:
                    outlier_pct = (outliers/total*100) if total > 0 else 0
                    st.metric("Outliers", f"{outliers:,} ({outlier_pct:.1f}%)")
                
                st.markdown("---")
                st.subheader("📈 Lead Time Trend Over Time")
                
                monthly = df.groupby(["month_start", "month_name"])["lead_time_calc"].agg(['mean', 'median']).reset_index()
                monthly = monthly.sort_values("month_start")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["mean"], name="Average", mode="lines+markers", line=dict(width=3)))
                fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["median"], name="Median", mode="lines+markers", line=dict(dash="dash")))
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏙️ Top 15 Cities")
                    if config.get("city_col"):
                        top_city = df.groupby(config["city_col"]).agg({"lead_time_calc": "mean", config["id_col"]: "count"}).reset_index()
                        top_city.columns = ["City", "Avg LT", "Volume"]
                        top_city = top_city.sort_values("Volume", ascending=False).head(15)
                        fig_city = px.bar(top_city, y="City", x="Volume", orientation="h", text="Volume", color="Avg LT", color_continuous_scale="RdYlGn_r")
                        fig_city.update_traces(textposition="outside")
                        fig_city.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig_city, use_container_width=True)
                    else:
                        st.info("City column not mapped")
                
                with col2:
                    st.subheader("🚗 Top 15 Models")
                    top_model = df.groupby(config["model_col"]).agg({"lead_time_calc": "mean", config["id_col"]: "count"}).reset_index()
                    top_model.columns = ["Model", "Avg LT", "Volume"]
                    top_model = top_model.sort_values("Volume", ascending=False).head(15)
                    fig_model = px.bar(top_model, y="Model", x="Volume", orientation="h", text="Volume", color="Avg LT", color_continuous_scale="RdYlGn_r")
                    fig_model.update_traces(textposition="outside")
                    fig_model.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig_model, use_container_width=True)
            
            # ===== TAB 2: PERFORMANCE =====
            with tab2:
                st.header("🎯 Performance Analysis")
                
                # Dynamic group by selector
                group_options = []
                group_map = {}
                
                if config.get("province_col"):
                    group_options.append("Province")
                    group_map["Province"] = config["province_col"]
                
                group_options.append("Model")
                group_map["Model"] = config["model_col"]
                
                if config.get("transporter_col"):
                    group_options.append("Transporter")
                    group_map["Transporter"] = config["transporter_col"]
                
                if config.get("city_col"):
                    group_options.append("City")
                    group_map["City"] = config["city_col"]
                
                selected_group = st.selectbox("Group By", options=group_options)
                group_col = group_map[selected_group]
                
                # Aggregate data
                perf_data = df.groupby(group_col).agg({
                    "lead_time_calc": ["mean", "median", "std"],
                    config["id_col"]: "count",
                    "is_outlier": "sum"
                }).reset_index()
                perf_data.columns = [selected_group, "Avg LT", "Median LT", "Std Dev", "Volume", "Outliers"]
                perf_data["Outlier %"] = (perf_data["Outliers"] / perf_data["Volume"] * 100).round(1)
                perf_data = perf_data.sort_values("Avg LT", ascending=False).head(20)
                
                # Chart
                fig = px.bar(
                    perf_data, 
                    y=selected_group, 
                    x="Avg LT", 
                    orientation="h", 
                    text="Avg LT", 
                    color="Avg LT", 
                    color_continuous_scale="RdYlGn_r",
                    hover_data=["Volume", "Outlier %"]
                )
                fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Detailed table
                st.subheader(f"📊 Detailed Statistics - {selected_group}")
                st.dataframe(
                    perf_data.style.background_gradient(cmap="RdYlGn_r", subset=["Avg LT"])
                                   .background_gradient(cmap="Reds", subset=["Outlier %"])
                                   .format({"Avg LT": "{:.1f}", "Median LT": "{:.1f}", "Std Dev": "{:.1f}", "Outlier %": "{:.1f}%"}),
                    use_container_width=True, hide_index=True
                )
            
            # ===== TAB 3: OUTLIER =====
            with tab3:
                st.header("🚨 Outlier Analysis")
                
                outliers_df = df[df["is_outlier"]].copy()
                
                if len(outliers_df) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Outliers", f"{len(outliers_df):,}")
                    with col2:
                        st.metric("Avg Lead Time", f"{outliers_df['lead_time_calc'].mean():.1f} days")
                    with col3:
                        st.metric("Max Lead Time", f"{outliers_df['lead_time_calc'].max():.0f} days")
                    
                    st.markdown("---")
                    
                    # Simplified filters: Model and City only
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
                    
                    # Apply filters
                    outliers_filtered = outliers_df[outliers_df[config["model_col"]].isin(model_filter)].copy()
                    if city_filter and config.get("city_col"):
                        outliers_filtered = outliers_filtered[outliers_filtered[config["city_col"]].isin(city_filter)]
                    
                    st.info(f"📊 Showing {len(outliers_filtered):,} of {len(outliers_df):,} outliers")
                    
                    st.markdown("---")
                    
                    # Show ALL columns for filtered outliers
                    st.subheader("📋 Complete Outlier Records")
                    
                    # Sort by lead time descending
                    outliers_display = outliers_filtered.sort_values("lead_time_calc", ascending=False)
                    
                    # Display all columns from original data
                    st.dataframe(outliers_display, use_container_width=True, height=500)
                    
                    # Download filtered outliers
                    csv = outliers_display.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Filtered Outliers", csv, f"outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                    
                    st.markdown("---")
                    
                    # Summary charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🚗 Top 10 Models with Outliers")
                        top_model_outlier = outliers_filtered[config["model_col"]].value_counts().head(10)
                        fig = px.bar(x=top_model_outlier.values, y=top_model_outlier.index, orientation="h", text=top_model_outlier.values)
                        fig.update_traces(marker_color='#dc3545', textposition="outside")
                        fig.update_layout(height=400, showlegend=False, yaxis_title="Model", xaxis_title="Outlier Count")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if config.get("city_col"):
                            st.subheader("🏙️ Top 10 Cities with Outliers")
                            top_city_outlier = outliers_filtered[config["city_col"]].value_counts().head(10)
                            fig = px.bar(x=top_city_outlier.values, y=top_city_outlier.index, orientation="h", text=top_city_outlier.values)
                            fig.update_traces(marker_color='#ffc107', textposition="outside")
                            fig.update_layout(height=400, showlegend=False, yaxis_title="City", xaxis_title="Outlier Count")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("🎉 No outliers detected!")
        
        else:
            st.info("Click 'Analyze Data' in sidebar to start")
            st.subheader("📋 Data Preview")
            st.dataframe(df_raw.head(20), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    st.info("Upload your Excel/CSV file to begin")
    st.markdown("""
    ### Required Columns:
    - **VIN** or ID (unique vehicle identifier)
    - **TOS Date (Start)** **or** **Date ETD** (fallback)
    - **ATA Date (End)** **or** **Date ATA** (fallback)
    - **Outlet/PDC** (dealer/destination)
    - **Model** (vehicle model)
    
    ### Optional but Recommended:
    - **City** (destination city)
    - **Province** (destination province)
    - **Region** (geographic region)
    - **Transporter** (carrier company)
    - **TOS** (type of service/mode)
    - **Date ATD** (actual departure)
    - **Date ETA** (estimated arrival)
    
    ### Example Data:
    """)
    
    example = pd.DataFrame({
        'VIN': ['KMHXX00XXXX000001', 'KMHXX00XXXX000002', 'KMHXX00XXXX000003'],
        'Model': ['CRETA', 'STARGAZER', 'IONIQ 5'],
        'TOS': ['Darat', 'Laut', 'Laut'],
        'TOS Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'ATA Date': ['2024-01-05', '2024-01-12', '2024-01-15'],
        'Destination': ['Jakarta', 'Surabaya', 'Medan'],
        'City': ['Jakarta', 'Surabaya', 'Medan'],
        'Region': ['Jawa', 'Jawa', 'Sumatra'],
        'Outlet/PDC': ['Dealer A', 'Dealer B', 'Dealer C'],
        'Transporter': ['Transporter X', 'Transporter Y', 'Transporter Z']
    })
    
    st.dataframe(example, use_container_width=True)
    
    csv_example = example.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Example Template",
        csv_example,
        'template_hyundai_distribution.csv',
        'text/csv'
    )
