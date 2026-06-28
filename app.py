import streamlit as st
import pandas as pd
import time
from datetime import datetime
import pydeck as pdk

# ShieldFlow Imports
from shieldflow.engine import ShieldFlowEngine
from scripts.generator import (
    generate_cold_start,
    generate_model_activation,
    generate_impossible_speed,
    generate_device_flag,
    generate_outlier_flag,
    generate_adaptive_cluster
)

# Page Configuration
st.set_page_config(
    page_title="ShieldFlow Spatiotemporal Sandbox",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #080c14;
    }
    .metric-card {
        background-color: rgba(13, 20, 35, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .console-log {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background-color: #04060b;
        color: #60a5fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        height: 250px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "engine" not in st.session_state:
    st.session_state.engine = ShieldFlowEngine()
    
if "logs" not in st.session_state:
    st.session_state.logs = ["[SYSTEM] Sandbox initialized. Select a scenario in the sidebar to begin."]
    
if "pings" not in st.session_state:
    st.session_state.pings = []  # list of dicts with lat, lon, verdict, type
    
if "raw_stream" not in st.session_state:
    st.session_state.raw_stream = []

# Sidebar Controls
st.sidebar.title("🛡️ ShieldFlow Control Panel")
st.sidebar.markdown("Execute spatiotemporal security scenarios to test the ShieldFlow microservice plugin.")

speed_delay = st.sidebar.slider("Simulation Speed Delay (s)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

st.sidebar.subheader("Security Scenarios")

def run_simulation(scenario_name: str, generator_fn):
    # Reset engine and clean state for a deterministic scenario demonstration
    st.session_state.engine.store.clear()
    st.session_state.engine.retrain_counts.clear()
    st.session_state.logs = [f"[SYSTEM] Starting scenario: {scenario_name}..."]
    st.session_state.pings = []
    st.session_state.raw_stream = []
    
    # Generate events
    events = generator_fn()
    
    # Setup progress placeholders
    progress_bar = st.sidebar.progress(0)
    
    # Loop and stream
    for idx, event in enumerate(events):
        st.session_state.raw_stream.append(event)
        
        # Evaluate Risk via the engine
        res = st.session_state.engine.evaluate_risk(event)
        
        # Extract verdict info
        verdict = res["verdict"]
        is_flagged = res["is_flagged"]
        state = res["telemetry"]["engine_state"]
        speed = res["telemetry"]["calculated_velocity_kmh"]
        spatial = res["telemetry"]["spatial_status"]
        
        ts_str = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
        
        log_entry = f"[{ts_str}] User: {event['user_id'][:8]}... | Status: {verdict} | State: {state} | Speed: {speed} km/h | Spatial: {spatial} | Flagged: {is_flagged}"
        st.session_state.logs.append(log_entry)
        
        # If flagged, simulate step-up verification (user resolves MFA challenge)
        if is_flagged:
            st.session_state.logs.append(f"  ↳ [MFA] ⚠️ Step-up Challenge prompted... Verified successfully.")
            st.session_state.engine.verify_login(event)
            st.session_state.logs.append(f"  ↳ [ENGINE] Login verified & registered. Retraining checked.")
            
        # Record coordinates for map
        color = [16, 185, 129]  # Green
        if verdict == "MEDIUM_RISK":
            color = [245, 158, 11]  # Orange
        elif verdict == "HIGH_RISK":
            color = [239, 68, 68]  # Red
            
        st.session_state.pings.append({
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "verdict": verdict,
            "color": color
        })
        
        # Update UI dynamically
        progress_bar.progress((idx + 1) / len(events))
        time.sleep(speed_delay)
        st.rerun()

# Scenario Buttons
if st.sidebar.button("Simulate Cold Start Lifecycle", use_container_width=True):
    run_simulation("Cold Start Lifecycle", generate_cold_start)

if st.sidebar.button("Trigger Baseline Cluster Formation", use_container_width=True):
    run_simulation("Baseline Cluster Formation", generate_model_activation)

if st.sidebar.button("Execute Impossible Speed Intrusion", use_container_width=True):
    run_simulation("Impossible Speed Intrusion", generate_impossible_speed)

if st.sidebar.button("Simulate Device Fingerprint Mutation", use_container_width=True):
    run_simulation("Device Fingerprint Mutation", generate_device_flag)

if st.sidebar.button("Inject Isolated Spatial Outlier", use_container_width=True):
    run_simulation("Isolated Spatial Outlier", generate_outlier_flag)

if st.sidebar.button("Demonstrate Relocation & Adaptation", use_container_width=True):
    run_simulation("Relocation & Adaptation", generate_adaptive_cluster)

if st.sidebar.button("Reset Engine Store", type="primary", use_container_width=True):
    st.session_state.engine.store.clear()
    st.session_state.engine.retrain_counts.clear()
    st.session_state.logs = ["[SYSTEM] Engine store cleared successfully."]
    st.session_state.pings = []
    st.session_state.raw_stream = []
    st.success("State cleared.")

# Main Application Layout
st.title("🛡️ ShieldFlow Sandbox & Profiler")
st.markdown("Real-time Spatiotemporal Risk Evaluation Microservice Sandbox.")

# Count Risk Categories
low_count = sum(1 for p in st.session_state.pings if p["verdict"] == "LOW_RISK")
med_count = sum(1 for p in st.session_state.pings if p["verdict"] == "MEDIUM_RISK")
high_count = sum(1 for p in st.session_state.pings if p["verdict"] == "HIGH_RISK")

# Metrics Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Low Risk Logins (Allowed)", low_count)
with col2:
    st.metric("Medium Risk Logins (MFA)", med_count)
with col3:
    st.metric("High Risk Logins (Flagged)", high_count)

# Interactive Map & Console Logs
m_col, l_col = st.columns([3, 2])

with m_col:
    st.subheader("🛰️ Spatiotemporal Map Visualizer")
    
    # Create Pings Dataframe
    pings_df = pd.DataFrame(st.session_state.pings)
    
    # Create Clusters Dataframe
    cluster_records = []
    for user_id in st.session_state.engine.store._profiles:
        profile_clusters = st.session_state.engine.store.get_profile(user_id)
        for c in profile_clusters:
            cluster_records.append({
                "latitude": c["centroid_lat"],
                "longitude": c["centroid_lon"],
                "radius_meters": c["dynamic_radius"] * 1000
            })
    clusters_df = pd.DataFrame(cluster_records)
    
    # Layer Setup
    layers = []
    
    if not clusters_df.empty:
        # Show safe cluster geofence zones
        clusters_layer = pdk.Layer(
            "ScatterplotLayer",
            clusters_df,
            get_position="[longitude, latitude]",
            get_color="[16, 185, 129, 60]",  # Light green transparent geofence
            get_radius="radius_meters",
            pickable=True
        )
        layers.append(clusters_layer)
        
    if not pings_df.empty:
        # Show actual login coordinates
        pings_layer = pdk.Layer(
            "ScatterplotLayer",
            pings_df,
            get_position="[longitude, latitude]",
            get_color="color",
            get_radius=15000,
            pickable=True
        )
        layers.append(pings_layer)
        
    # Map rendering
    initial_lat = pings_df["latitude"].mean() if not pings_df.empty else 37.7749
    initial_lon = pings_df["longitude"].mean() if not pings_df.empty else -122.4194
    initial_zoom = 3 if not pings_df.empty else 1
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=initial_lat,
            longitude=initial_lon,
            zoom=initial_zoom,
            pitch=0
        ),
        tooltip={"text": "Verdict: {verdict}"} if not pings_df.empty else None
    )
    st.pydeck_chart(deck)

with l_col:
    st.subheader("📟 System Telemetry Console")
    # Display log records
    log_text = "\n".join(st.session_state.logs)
    st.text_area("Console Stream Output", log_text, height=450, disabled=True)

# Display User Profiles
st.subheader("👤 User Profiles & Dynamic Clusters")
profile_keys = list(st.session_state.engine.store._profiles.keys())
if profile_keys:
    for uid in profile_keys:
        profile_clusters = st.session_state.engine.store.get_profile(uid)
        rounds = st.session_state.engine.retrain_counts.get(uid, 0)
        st.write(f"**User ID:** `{uid}` | **Retrain Rounds:** `{rounds}`")
        st.json(profile_clusters)
else:
    st.info("No spatial profiles computed yet. Retraining executes once user history reaches 10 logins.")

# Display Raw JSON logs
with st.expander("Incoming Device Stream (Raw JSON Logs)"):
    st.json(st.session_state.raw_stream)
