"""ShieldFlow FastAPI Gatekeeper API.

Exposes endpoints to evaluate real-time login risks and verify anomaly MFA challenges.
Also hosts the built-in HTML/CSS/JS interactive Hugging Face Spaces prototype checker.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import math

from ml.features.geo_features import spatiotemporal_velocity
from ml.models.registry import ModelRegistry
from ml.models.inference_ops import evaluate_login_location
from ml.training.retrain import check_and_trigger_retrain
from algorithms.graph.velocity_validator import validate_velocity
from algorithms.ranking.multi_factor_score import calculate_risk_score
from algorithms.spatial.haversine import haversine_distance

from app.services import RedisClient, KafkaProducer, PostgresClient

app = FastAPI(
    title="ShieldFlow Risk Evaluation Gateway",
    description="Deterministic Graph + Spatial DBSCAN Risk Evaluator API",
    version="1.0.0"
)

# Initialize service layer clients
redis_client = RedisClient()
kafka_producer = KafkaProducer()
db_client = PostgresClient()
registry = ModelRegistry()

# --- Pydantic Data Models ---
class LoginEvent(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", example="user_123")
    latitude: float = Field(..., description="Login latitude coordinate", example=37.7749)
    longitude: float = Field(..., description="Login longitude coordinate", example=-122.4194)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp(), description="UNIX timestamp")
    device_hash: str = Field(..., description="Client device unique hash value", example="abc123device")

class AnomalyVerification(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    latitude: float = Field(..., description="Login latitude coordinate")
    longitude: float = Field(..., description="Login longitude coordinate")
    timestamp: float = Field(..., description="UNIX timestamp")
    device_hash: str = Field(..., description="Client device unique hash value")
    is_verified: bool = Field(..., description="Did the user successfully pass MFA challenge?")

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/evaluate_risk")
def evaluate_risk(event: LoginEvent):
    """Evaluates the risk level of an incoming login event using Graph + ML."""
    user_id = event.user_id
    lat = event.latitude
    lon = event.longitude
    ts = event.timestamp
    device = event.device_hash
    
    # 1. Fetch last verified login node (from Redis cache or Postgres)
    last_node = redis_client.get_last_node(user_id)
    velocity_kmh = 0.0
    
    # 2. Graph Validator: Spatiotemporal Velocity Check
    if last_node:
        # Check velocity
        velocity_kmh = spatiotemporal_velocity(
            last_node["latitude"], last_node["longitude"], last_node["timestamp"],
            lat, lon, ts
        )
        is_possible = validate_velocity(last_node, {"latitude": lat, "longitude": lon, "timestamp": ts})
        
        if not is_possible:
            verdict = "HIGH_RISK"
            status = "IMPOSSIBLE_VELOCITY"
            score = 1.0
            reason = f"Impossible velocity of {velocity_kmh:.2f} km/h detected relative to last login."
            
            # Emit Kafka Alert immediately
            kafka_producer.emit_event(
                topic="shieldflow.anomalies",
                key=user_id,
                value={"user_id": user_id, "reason": reason, "velocity_kmh": velocity_kmh, "verdict": verdict}
            )
            return {
                "verdict": verdict,
                "status": status,
                "score": score,
                "reason": reason,
                "details": {"velocity_kmh": round(velocity_kmh, 2), "distance_km": None}
            }
            
    # 3. Model Registry Lookup: Spatial Profile Checks
    profile = registry.get_profile(user_id)
    
    # 3a. Cold Start Fallback
    if not profile:
        verdict = "LOW_RISK"
        status = "COLD_START_BYPASS"
        score = 0.0
        reason = "User is in Cold Start state (< 10 logins). Passed velocity checks."
        
        # In cold start, if it is low risk, we auto-verify to build baseline
        db_client.record_login(user_id, lat, lon, ts, device, is_verified=True)
        redis_client.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
        
        return {
            "verdict": verdict,
            "status": status,
            "score": score,
            "reason": reason,
            "details": {"velocity_kmh": round(velocity_kmh, 2), "distance_km": None}
        }
        
    # 3b. ML Inference boundary check
    evaluation = evaluate_login_location(user_id, lat, lon)
    
    if evaluation["status"] == "KNOWN_ZONE":
        verdict = "LOW_RISK"
        status = "KNOWN_ZONE"
        score = 0.0
        reason = f"Login matches spatial cluster {evaluation['cluster_idx']}."
        
        # Save low risk as verified login
        db_client.record_login(user_id, lat, lon, ts, device, is_verified=True)
        redis_client.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
        
        return {
            "verdict": verdict,
            "status": status,
            "score": score,
            "reason": reason,
            "details": {
                "velocity_kmh": round(velocity_kmh, 2),
                "distance_km": round(evaluation["distance_km"], 2) if "distance_km" in evaluation else 0.0
            }
        }
        
    # 3c. Spatial Outlier: Fallback to Multi-Factor Scoring
    # Determine distance from closest centroid
    clusters = profile.get("clusters", [])
    closest_dist = 99999.0
    for cluster in clusters:
        dist = haversine_distance(lat, lon, cluster["centroid_lat"], cluster["centroid_lon"])
        if dist < closest_dist:
            closest_dist = dist
            
    # Check device mismatch
    historic_devices = db_client.get_all_device_hashes(user_id)
    device_mismatch = device not in historic_devices if historic_devices else True
    
    # Calculate Multi-Factor score
    mfa_score = calculate_risk_score(velocity_kmh, closest_dist, device_mismatch)
    
    # Verdict threshold check
    if mfa_score >= 0.70:
        verdict = "HIGH_RISK"
        reason = f"Spatial outlier with high multi-factor risk score of {mfa_score:.2f}."
    else:
        verdict = "MEDIUM_RISK"
        reason = f"Spatial outlier with moderate multi-factor risk score of {mfa_score:.2f}."
        
    status = "OUTLIER"
    
    # Emit Kafka event for verification pipeline
    kafka_producer.emit_event(
        topic="shieldflow.anomalies",
        key=user_id,
        value={
            "user_id": user_id,
            "reason": reason,
            "score": mfa_score,
            "velocity_kmh": velocity_kmh,
            "distance_km": closest_dist,
            "device_mismatch": device_mismatch,
            "verdict": verdict
        }
    )
    
    return {
        "verdict": verdict,
        "status": status,
        "score": mfa_score,
        "reason": reason,
        "details": {
            "velocity_kmh": round(velocity_kmh, 2),
            "distance_km": round(closest_dist, 2),
            "device_mismatch": device_mismatch
        }
    }

@app.post("/verify_anomaly")
def verify_anomaly(payload: AnomalyVerification, background_tasks: BackgroundTasks):
    """Processes MFA outcome to update baselines and trigger retraining checks."""
    user_id = payload.user_id
    lat = payload.latitude
    lon = payload.longitude
    ts = payload.timestamp
    device = payload.device_hash
    is_verified = payload.is_verified
    
    # 1. Record the outcome in DB
    db_client.record_login(user_id, lat, lon, ts, device, is_verified)
    
    # 2. Update cache if verified
    if is_verified:
        redis_client.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
        
    # 3. Trigger Retraining Engine check as a background task
    retrain_verdict = {}
    try:
        retrain_verdict = check_and_trigger_retrain(
            user_id=user_id,
            new_login_lat=lat,
            new_login_lon=lon,
            new_login_ts=ts,
            new_login_device=device
        )
    except Exception as e:
        print(f"[RETRAIN FAILED] Retrain logic check failed: {e}")
        
    return {
        "status": "success",
        "recorded_verified": is_verified,
        "retrain_status": retrain_verdict
    }

# --- Sample CSV Data Endpoint ---
@app.get("/sample_pings.csv")
def get_sample_pings():
    """Serves the sample spatiotemporal simulation CSV file."""
    from fastapi.responses import FileResponse
    return FileResponse("/home/dhruv/Documents/fraud-detector/data/sample_pings.csv", media_type="text/csv", filename="sample_pings.csv")

# --- Built-in Hugging Face Spaces Interactive UI ---
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the premium, dynamic web interface with Map-based simulation checker."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ShieldFlow Spatiotemporal Simulation Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <!-- Leaflet Map CSS -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            :root {
                --bg-color: #080c14;
                --card-bg: rgba(13, 20, 35, 0.75);
                --accent-blue: #3b82f6;
                --accent-green: #10b981;
                --accent-yellow: #f59e0b;
                --accent-red: #ef4444;
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --border-color: rgba(255, 255, 255, 0.08);
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg-color);
                background-image: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(239, 68, 68, 0.08) 0%, transparent 40%);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
            }
            header {
                padding: 1.2rem 2.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border-color);
                backdrop-filter: blur(12px);
                background: rgba(8, 12, 20, 0.5);
                z-index: 10;
            }
            header h1 {
                font-size: 1.5rem;
                font-weight: 800;
                background: linear-gradient(to right, #60a5fa, #ef4444);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
            }
            .status-badge {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent-green);
                padding: 0.4rem 0.8rem;
                border-radius: 20px;
                border: 1px solid rgba(16, 185, 129, 0.3);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .status-pulse {
                width: 8px;
                height: 8px;
                background: var(--accent-green);
                border-radius: 50%;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
            main {
                flex: 1;
                max-width: 1700px;
                width: 100%;
                margin: 0 auto;
                padding: 1.5rem 2.5rem;
                display: grid;
                grid-template-columns: 460px 1fr;
                gap: 1.5rem;
            }
            .panel {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                max-height: calc(100vh - 120px);
                overflow-y: auto;
                padding-right: 0.2rem;
            }
            /* Custom Scrollbar for panel */
            .panel::-webkit-scrollbar {
                width: 6px;
            }
            .panel::-webkit-scrollbar-track {
                background: transparent;
            }
            .panel::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(16px);
                box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3);
            }
            h2 {
                font-size: 1.15rem;
                margin-bottom: 1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .form-group {
                margin-bottom: 1rem;
            }
            label {
                display: block;
                font-size: 0.8rem;
                color: var(--text-muted);
                margin-bottom: 0.3rem;
                font-weight: 500;
            }
            input {
                width: 100%;
                padding: 0.7rem 0.9rem;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                color: var(--text-main);
                font-family: inherit;
                font-size: 0.9rem;
                transition: all 0.2s ease;
            }
            input:focus {
                outline: none;
                border-color: var(--accent-blue);
                background: rgba(255, 255, 255, 0.06);
            }
            .btn-primary {
                background: var(--accent-blue);
                color: white;
                border: none;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .btn-primary:hover {
                background: #2563eb;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
            }
            .btn-simulator {
                background: linear-gradient(135deg, var(--accent-blue), #4f46e5);
                color: white;
                font-weight: 700;
                font-size: 0.95rem;
                border: none;
                width: 100%;
                padding: 0.9rem;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.6rem;
                box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2);
            }
            .btn-simulator:hover {
                filter: brightness(1.1);
                transform: translateY(-1px);
            }
            .btn-simulator:disabled {
                background: #1f2937;
                color: #4b5563;
                cursor: not-allowed;
                box-shadow: none;
                filter: none;
            }
            .preset-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            .preset-btn {
                padding: 0.5rem;
                font-size: 0.8rem;
                border-radius: 6px;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--border-color);
                color: var(--text-muted);
                cursor: pointer;
                transition: all 0.2s;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.3rem;
            }
            .preset-btn:hover {
                background: rgba(255, 255, 255, 0.06);
                color: var(--text-main);
            }
            #map {
                width: 100%;
                height: 520px;
                border-radius: 12px;
                border: 1px solid var(--border-color);
                z-index: 1;
            }
            .map-container {
                display: flex;
                flex-direction: column;
                gap: 1.2rem;
            }
            .console-card {
                background: #04060b;
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 1rem;
                height: 220px;
                display: flex;
                flex-direction: column;
            }
            .console-header {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                color: var(--text-muted);
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                display: flex;
                justify-content: space-between;
            }
            .console-logs {
                flex: 1;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.8rem;
                overflow-y: auto;
                padding-top: 0.6rem;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }
            .log-item {
                line-height: 1.4;
            }
            .log-time { color: #6b7280; }
            .log-info { color: #60a5fa; }
            .log-success { color: var(--accent-green); }
            .log-warning { color: var(--accent-yellow); }
            .log-error { color: var(--accent-red); }
            
            .verdict-box {
                font-size: 1.6rem;
                font-weight: 800;
                padding: 0.8rem;
                border-radius: 8px;
                margin-top: 0.8rem;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .verdict-low {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid var(--accent-green);
                color: var(--accent-green);
            }
            .verdict-medium {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid var(--accent-yellow);
                color: var(--accent-yellow);
            }
            .verdict-high {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid var(--accent-red);
                color: var(--accent-red);
            }
            
            /* Custom Leaflet Dark Mode styling overrides */
            .leaflet-container {
                background: #0f172a !important;
            }
            .leaflet-tile {
                filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%);
            }
            .leaflet-popup-content-wrapper, .leaflet-popup-tip {
                background: #1e293b !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border-color);
                font-family: 'Outfit', sans-serif;
                font-size: 0.85rem;
            }
            .slider-container {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .slider-container input[type="range"] {
                flex: 1;
                cursor: pointer;
            }
            .slider-value {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                min-width: 45px;
                text-align: right;
            }
            .csv-status-card {
                background: rgba(255, 255, 255, 0.02);
                border: 1px dashed var(--border-color);
                border-radius: 8px;
                padding: 0.8rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
            }
            .csv-status-title {
                font-size: 0.8rem;
                font-weight: 600;
            }
            .csv-status-detail {
                font-size: 0.75rem;
                color: var(--text-muted);
                line-height: 1.3;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>ShieldFlow Spatiotemporal Simulation</h1>
            <div class="status-badge">
                <div class="status-pulse"></div>
                GATEWAY RUNNING
            </div>
        </header>
        
        <main>
            <div class="panel">
                <!-- CSV Simulation Control -->
                <div class="card">
                    <h2>📁 CSV Spatiotemporal Simulation</h2>
                    <div style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.8rem; line-height: 1.4;">
                        Upload a CSV containing location logs or load our preconfigured multi-factor scenario (impossible speed, spatial clustering outliers, device mismatches).
                    </div>
                    
                    <div class="preset-grid">
                        <button class="preset-btn" onclick="loadSampleCsv()">
                            📥 Load Sample CSV
                        </button>
                        <button class="preset-btn" onclick="triggerFileInput()">
                            📁 Upload Custom CSV
                        </button>
                        <input type="file" id="csvFileInput" accept=".csv" style="display:none;" onchange="handleCsvUpload(event)">
                    </div>
                    
                    <div class="csv-status-card" id="csvStatusCard">
                        <div class="csv-status-title" id="csvStatusTitle">No CSV loaded</div>
                        <div class="csv-status-detail" id="csvStatusDetail">Ready to simulate. Load the sample dataset or select your own formatted CSV.</div>
                    </div>
                    
                    <div class="slider-container">
                        <label for="speedSlider" style="margin-bottom:0;">Sim Speed:</label>
                        <input type="range" id="speedSlider" min="0.5" max="5.0" step="0.5" value="2.0">
                        <span class="slider-value" id="speedValue">2.0s</span>
                    </div>
                    
                    <button class="btn-simulator" id="runCsvSimBtn" onclick="runCsvSimulation()" disabled>
                        🚀 Run CSV Simulation
                    </button>
                </div>
                
                <!-- Legacy Demo Simulation Control -->
                <div class="card">
                    <h2>🎯 Global Default Simulator</h2>
                    <div style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.8rem; line-height: 1.4;">
                        Runs a hardcoded multi-continent login path (SF -> Oakland -> LA -> London -> Paris) to verify core spatiotemporal algorithms.
                    </div>
                    <button class="preset-btn" style="width:100%; padding: 0.8rem; font-weight: 600; border-color: var(--accent-blue); color: var(--text-main);" onclick="startGlobalSimulation()">
                        Run Global Route Demo
                    </button>
                </div>
                
                <!-- Manual Live Evaluation -->
                <div class="card">
                    <h2>⚡ Live Evaluation Checker</h2>
                    <div class="preset-grid">
                        <button class="preset-btn" onclick="loadPreset('sf')">SF (Home)</button>
                        <button class="preset-btn" onclick="loadPreset('la_normal')">LA (Normal Travel)</button>
                        <button class="preset-btn" onclick="loadPreset('london_fast')">London (Impossible)</button>
                        <button class="preset-btn" onclick="loadPreset('paris_mfa')">Paris (MFA Pass)</button>
                    </div>
                    
                    <div class="form-group">
                        <label for="userId">User ID</label>
                        <input type="text" id="userId" value="4db924e8-0522-462c-865d-513f56d4f1bb">
                    </div>
                    <div class="form-group">
                        <label for="latitude">Latitude</label>
                        <input type="number" step="0.0001" id="latitude" value="19.0218">
                    </div>
                    <div class="form-group">
                        <label for="longitude">Longitude</label>
                        <input type="number" step="0.0001" id="longitude" value="72.8726">
                    </div>
                    <div class="form-group">
                        <label for="deviceHash">Device Hash</label>
                        <input type="text" id="deviceHash" value="dev_mumbai_iphone">
                    </div>
                    <button class="btn-primary" style="width:100%; padding: 0.8rem; border-radius:8px; border:none; font-weight:600;" onclick="submitManualCheck()">
                        Evaluate Single Login
                    </button>
                    
                    <div id="verdictDisplay" style="display:none;">
                        <div class="verdict-box" id="verdictBox">LOW RISK</div>
                    </div>
                </div>
            </div>
            
            <div class="map-container">
                <div class="card" style="padding: 1rem; flex: 1;">
                    <div id="map"></div>
                </div>
                
                <div class="console-card">
                    <div class="console-header">
                        <span>🛰️ SHIELDFLOW REAL-TIME TELEMETRY LOGS</span>
                        <span>SLA Target: &lt;50ms</span>
                    </div>
                    <div class="console-logs" id="consoleLogs">
                        <div class="log-item"><span class="log-time">[SYSTEM]</span> <span class="log-info">Dashboard initialized. Select simulation flow to begin.</span></div>
                    </div>
                </div>
            </div>
        </main>
        
        <!-- Leaflet Map JS -->
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        
        <script>
            // Initialize Leaflet Map centered on global view
            const map = L.map('map').setView([20, 0], 2);
            
            // OpenStreetMap tile layer (inverted to dark mode via CSS filter)
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 18,
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
            
            let markerGroup = L.layerGroup().addTo(map);
            let polylineGroup = L.layerGroup().addTo(map);
            
            // Speed Slider update
            const speedSlider = document.getElementById('speedSlider');
            const speedValue = document.getElementById('speedValue');
            speedSlider.oninput = function() {
                speedValue.textContent = this.value + 's';
            }

            // Holds parsed CSV rows
            let loadedPings = [];

            // Custom Icons for Risk Verdicts
            function createCircleMarker(lat, lon, verdict) {
                let color = '#3b82f6'; // blue (default)
                if (verdict === 'LOW_RISK') color = '#10b981'; // green
                if (verdict === 'MEDIUM_RISK') color = '#f59e0b'; // orange
                if (verdict === 'HIGH_RISK') color = '#ef4444'; // red
                
                return L.circleMarker([lat, lon], {
                    radius: 9,
                    fillColor: color,
                    color: '#ffffff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85
                });
            }

            const USER_PRESETS = {
                sf: { lat: 19.0218, lon: 72.8726, device: "dev_mumbai_iphone", label: "Mumbai (Known)" },
                la_normal: { lat: 19.0320, lon: 72.8101, device: "dev_mumbai_iphone", label: "Mumbai Outskirts" },
                london_fast: { lat: 40.7128, lon: -74.0060, device: "dev_mumbai_iphone", label: "New York (Impossible)" },
                paris_mfa: { lat: 18.5204, lon: 73.8567, device: "dev_unknown_macbook", label: "Pune (Outlier)" }
            };

            function loadPreset(key) {
                const p = USER_PRESETS[key];
                if (p) {
                    document.getElementById('latitude').value = p.lat;
                    document.getElementById('longitude').value = p.lon;
                    document.getElementById('deviceHash').value = p.device;
                    addLog(`Loaded preset parameters for ${p.label}`, "info");
                }
            }

            function addLog(text, type = "info") {
                const logs = document.getElementById('consoleLogs');
                const now = new Date();
                const timeStr = now.toTimeString().split(' ')[0];
                
                let typeClass = "log-info";
                if (type === "success") typeClass = "log-success";
                if (type === "warning") typeClass = "log-warning";
                if (type === "error") typeClass = "log-error";
                
                logs.innerHTML += `<div class="log-item"><span class="log-time">[${timeStr}]</span> <span class="${typeClass}">${text}</span></div>`;
                logs.scrollTop = logs.scrollHeight;
            }

            function triggerFileInput() {
                document.getElementById('csvFileInput').click();
            }

            // CSV Parsing Logic
            function parseCSV(text) {
                const lines = text.split(/\\r?\\n/);
                const result = [];
                if (lines.length < 2) return result;
                
                const headers = lines[0].split(',').map(h => h.trim());
                
                for (let i = 1; i < lines.length; i++) {
                    const line = lines[i].trim();
                    if (!line) continue;
                    
                    // Simple split respecting basic comma spacing
                    const values = line.split(',').map(v => v.trim());
                    const obj = {};
                    headers.forEach((header, idx) => {
                        obj[header] = values[idx] || "";
                    });
                    result.push(obj);
                }
                return result;
            }

            async function loadSampleCsv() {
                addLog("Fetching sample spatiotemporal CSV...", "info");
                try {
                    const response = await fetch('/sample_pings.csv');
                    const text = await response.text();
                    loadedPings = parseCSV(text);
                    
                    if (loadedPings.length > 0) {
                        document.getElementById('csvStatusTitle').textContent = "sample_pings.csv loaded";
                        document.getElementById('csvStatusDetail').textContent = `Parsed ${loadedPings.length} spatiotemporal login events correctly. Ready to run simulation.`;
                        document.getElementById('runCsvSimBtn').disabled = false;
                        addLog(`Loaded sample CSV with ${loadedPings.length} events. Ready to run simulation.`, "success");
                    } else {
                        throw new Error("Empty CSV output");
                    }
                } catch (err) {
                    addLog(`Failed to fetch sample CSV: ${err.message}`, "error");
                }
            }

            function handleCsvUpload(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    const text = e.target.result;
                    loadedPings = parseCSV(text);
                    
                    if (loadedPings.length > 0) {
                        document.getElementById('csvStatusTitle').textContent = file.name;
                        document.getElementById('csvStatusDetail').textContent = `Parsed ${loadedPings.length} custom login events. Ready to run simulation.`;
                        document.getElementById('runCsvSimBtn').disabled = false;
                        addLog(`Successfully parsed custom CSV "${file.name}" with ${loadedPings.length} login pings.`, "success");
                    } else {
                        addLog("Error: CSV file contains invalid structure or has no data rows.", "error");
                    }
                };
                reader.readAsText(file);
            }

            async function runCsvSimulation() {
                if (loadedPings.length === 0) return;
                
                // Clear map layers
                markerGroup.clearLayers();
                polylineGroup.clearLayers();
                
                addLog(`▶️ Starting CSV Spatiotemporal Simulation (${loadedPings.length} pings)...`, "info");
                
                // Disable controls
                document.getElementById('runCsvSimBtn').disabled = true;
                const delayMs = parseFloat(speedSlider.value) * 1000;
                
                let coordinatesPath = [];
                
                for (let i = 0; i < loadedPings.length; i++) {
                    const ping = loadedPings[i];
                    const locName = ping.location_name || `Coordinate #${i+1}`;
                    const desc = ping.description || "";
                    
                    addLog(`Ping ${i+1}/${loadedPings.length}: Evaluating ${locName}...`, "info");
                    
                    try {
                        const response = await fetch('/evaluate_risk', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: ping.user_id,
                                latitude: parseFloat(ping.latitude),
                                longitude: parseFloat(ping.longitude),
                                timestamp: parseFloat(ping.timestamp),
                                device_hash: ping.device_hash
                            })
                        });
                        
                        const data = await response.json();
                        
                        // Map marker rendering
                        const marker = createCircleMarker(parseFloat(ping.latitude), parseFloat(ping.longitude), data.verdict);
                        marker.addTo(markerGroup);
                        
                        let popupText = `<b>Ping ${i+1}: ${locName}</b><br>
                                         <b>Verdict:</b> ${data.verdict}<br>
                                         <b>Status:</b> ${data.status}<br>`;
                        if (data.details.velocity_kmh) {
                            popupText += `<b>Velocity:</b> ${data.details.velocity_kmh.toFixed(1)} km/h<br>`;
                        }
                        if (data.details.distance_km) {
                            popupText += `<b>Distance:</b> ${data.details.distance_km.toFixed(1)} km<br>`;
                        }
                        popupText += `<i>${data.reason}</i>`;
                        marker.bindPopup(popupText);
                        
                        let logType = "success";
                        if (data.verdict === 'MEDIUM_RISK') logType = "warning";
                        if (data.verdict === 'HIGH_RISK') logType = "error";
                        
                        // Detailed spatiotemporal logs based on validation status
                        if (data.status === 'IMPOSSIBLE_VELOCITY') {
                            addLog(`🔴 [IMPOSSIBLE_TRAVEL] User ${ping.user_id} flagged at ${locName}! Speed: ${data.details.velocity_kmh.toFixed(1)} km/h. Verdict: HIGH_RISK.`, "error");
                        } else if (data.status === 'OUTLIER') {
                            const badge = data.verdict === 'HIGH_RISK' ? '🔴' : '🟡';
                            addLog(`${badge} [SPATIAL_OUTLIER] User ${ping.user_id} evaluated at ${locName}. Dist: ${data.details.distance_km}km, Device Match: ${!data.details.device_mismatch}. Risk Score: ${data.score}. Verdict: ${data.verdict}.`, logType);
                        } else if (data.status === 'KNOWN_ZONE') {
                            addLog(`🟢 [KNOWN_ZONE] User ${ping.user_id} authenticated at ${locName}. Centroid Distance: ${data.details.distance_km || 0}km. Verdict: LOW_RISK.`, "success");
                        } else {
                            addLog(`🟢 [COLD_START] User ${ping.user_id} bypass check at ${locName}. Reason: ${data.reason}`, "success");
                        }
                        
                        // MFA Simulation block
                        if (data.verdict === 'MEDIUM_RISK' || data.verdict === 'HIGH_RISK') {
                            addLog(`⚠️ [MFA_CHALLENGE] Anomaly detected. Sending verification payload...`, "warning");
                            
                            const verifyResponse = await fetch('/verify_anomaly', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    user_id: ping.user_id,
                                    latitude: parseFloat(ping.latitude),
                                    longitude: parseFloat(ping.longitude),
                                    timestamp: parseFloat(ping.timestamp),
                                    device_hash: ping.device_hash,
                                    is_verified: true
                                })
                            });
                            const verifyData = await verifyResponse.json();
                            addLog(`✅ [MFA_SOLVED] MFA challenge passed. Base profile retraining status: ${verifyData.retrain_status.action || 'OK'}`, "success");
                        } else {
                            // Record standard log
                            await fetch('/verify_anomaly', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    user_id: ping.user_id,
                                    latitude: parseFloat(ping.latitude),
                                    longitude: parseFloat(ping.longitude),
                                    timestamp: parseFloat(ping.timestamp),
                                    device_hash: ping.device_hash,
                                    is_verified: true
                                })
                            });
                        }
                        
                        // Draw paths
                        coordinatesPath.push([parseFloat(ping.latitude), parseFloat(ping.longitude)]);
                        if (coordinatesPath.length > 1) {
                            let lineColor = '#3b82f6';
                            let dashArray = null;
                            if (data.verdict === 'HIGH_RISK') {
                                lineColor = '#ef4444';
                                dashArray = '8, 8';
                            } else if (data.verdict === 'MEDIUM_RISK') {
                                lineColor = '#f59e0b';
                            }
                            
                            L.polyline(coordinatesPath.slice(-2), {
                                color: lineColor,
                                weight: 3,
                                opacity: 0.7,
                                dashArray: dashArray
                            }).addTo(polylineGroup);
                        }
                        
                        // Pan to newest node and focus
                        map.panTo([parseFloat(ping.latitude), parseFloat(ping.longitude)]);
                        marker.openPopup();
                        
                    } catch (err) {
                        addLog(`Simulation Error at ping ${i+1}: ${err.message}`, "error");
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, delayMs));
                }
                
                // Final Fit Map
                if (coordinatesPath.length > 0) {
                    map.fitBounds(L.polyline(coordinatesPath).getBounds(), { padding: [50, 50] });
                }
                addLog(`🏁 CSV Simulation completed. Visualized ${loadedPings.length} sequential pings.`, "success");
                document.getElementById('runCsvSimBtn').disabled = false;
            }

            async function submitManualCheck() {
                const userId = document.getElementById('userId').value;
                const lat = parseFloat(document.getElementById('latitude').value);
                const lon = parseFloat(document.getElementById('longitude').value);
                const device = document.getElementById('deviceHash').value;
                
                addLog(`Checking risk for user ${userId} at [${lat}, ${lon}]...`, "info");
                
                try {
                    const response = await fetch('/evaluate_risk', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            latitude: lat,
                            longitude: lon,
                            device_hash: device
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Show verdict card
                    const display = document.getElementById('verdictDisplay');
                    const box = document.getElementById('verdictBox');
                    display.style.display = 'block';
                    box.textContent = data.verdict;
                    box.className = 'verdict-box';
                    
                    let logType = "success";
                    if (data.verdict === 'LOW_RISK') {
                        box.classList.add('verdict-low');
                    } else if (data.verdict === 'MEDIUM_RISK') {
                        box.classList.add('verdict-medium');
                        logType = "warning";
                    } else {
                        box.classList.add('verdict-high');
                        logType = "error";
                    }
                    
                    addLog(`Manual Check: ${data.verdict} | Status: ${data.status} | Reason: ${data.reason}`, logType);
                    
                    // Mark on map
                    markerGroup.clearLayers();
                    const marker = createCircleMarker(lat, lon, data.verdict);
                    marker.addTo(markerGroup);
                    marker.bindPopup(`<b>User:</b> ${userId}<br><b>Verdict:</b> ${data.verdict}<br><b>Reason:</b> ${data.reason}`).openPopup();
                    map.setView([lat, lon], 8);
                    
                } catch (err) {
                    addLog(`API request failed: ${err.message}`, "error");
                }
            }
            
            // Legacy simulation
            async function startGlobalSimulation() {
                markerGroup.clearLayers();
                polylineGroup.clearLayers();
                
                const simUser = "sim_user_" + Math.floor(Math.random() * 100000);
                addLog(`▶️ Started Global Route Demo for user: ${simUser}`, "info");
                
                const pings = [
                    { lat: 37.7749, lon: -122.4194, delayHours: 0, device: "iphone_sf", location: "San Francisco Home" },
                    { lat: 37.8044, lon: -122.2711, delayHours: 2, device: "iphone_sf", location: "Oakland Cafe" },
                    { lat: 34.0522, lon: -118.2437, delayHours: 8, device: "iphone_sf", location: "Los Angeles Office" },
                    { lat: 51.5074, lon: -0.1278, delayHours: 1, device: "iphone_sf", location: "London Airport (Impossible Travel)" },
                    { lat: 48.8566, lon: 2.3522, delayHours: 48, device: "macbook_paris", location: "Paris Hotel" }
                ];
                
                let baseTimestamp = 1600000000.0;
                let coordinatesPath = [];
                
                for (let i = 0; i < pings.length; i++) {
                    const ping = pings[i];
                    baseTimestamp += (ping.delayHours * 3600.0);
                    
                    addLog(`Evaluating Demo Ping ${i+1}/${pings.length} at ${ping.location}...`, "info");
                    
                    try {
                        const response = await fetch('/evaluate_risk', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: simUser,
                                latitude: ping.lat,
                                longitude: ping.lon,
                                timestamp: baseTimestamp,
                                device_hash: ping.device
                            })
                        });
                        
                        const data = await response.json();
                        const marker = createCircleMarker(ping.lat, ping.lon, data.verdict);
                        marker.addTo(markerGroup);
                        
                        let popupText = `<b>Ping ${i+1}: ${ping.location}</b><br>
                                         <b>Verdict:</b> ${data.verdict}<br>
                                         <b>Status:</b> ${data.status}<br>`;
                        if (data.details.velocity_kmh) {
                            popupText += `<b>Velocity:</b> ${data.details.velocity_kmh.toFixed(1)} km/h<br>`;
                        }
                        popupText += `<i>${data.reason}</i>`;
                        marker.bindPopup(popupText);
                        
                        let logType = "success";
                        if (data.verdict === 'MEDIUM_RISK') logType = "warning";
                        if (data.verdict === 'HIGH_RISK') logType = "error";
                        
                        addLog(`Demo Ping ${i+1}: ${data.verdict} (${data.status}) | Speed: ${data.details.velocity_kmh || 0} km/h | ${data.reason}`, logType);
                        
                        if (data.verdict === 'MEDIUM_RISK' || data.verdict === 'HIGH_RISK') {
                            addLog(`⚠️ Anomaly detected. Simulating MFA challenge solve...`, "warning");
                            const verifyResponse = await fetch('/verify_anomaly', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    user_id: simUser,
                                    latitude: ping.lat,
                                    longitude: ping.lon,
                                    timestamp: baseTimestamp,
                                    device_hash: ping.device,
                                    is_verified: true
                                })
                            });
                            const verifyData = await verifyResponse.json();
                            addLog(`✅ MFA challenge success. Profile retraining: ${JSON.stringify(verifyData.retrain_status.action)}`, "success");
                        } else {
                            await fetch('/verify_anomaly', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    user_id: simUser,
                                    latitude: ping.lat,
                                    longitude: ping.lon,
                                    timestamp: baseTimestamp,
                                    device_hash: ping.device,
                                    is_verified: true
                                })
                            });
                        }
                        
                        coordinatesPath.push([ping.lat, ping.lon]);
                        if (coordinatesPath.length > 1) {
                            let lineColor = '#3b82f6';
                            let dashArray = null;
                            if (data.verdict === 'HIGH_RISK') {
                                lineColor = '#ef4444';
                                dashArray = '8, 8';
                            }
                            L.polyline(coordinatesPath.slice(-2), {
                                color: lineColor,
                                weight: 3,
                                opacity: 0.7,
                                dashArray: dashArray
                            }).addTo(polylineGroup);
                        }
                        
                        map.panTo([ping.lat, ping.lon]);
                        marker.openPopup();
                        
                    } catch (err) {
                        addLog(`Demo Simulation Error: ${err.message}`, "error");
                    }
                    
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
                
                if (coordinatesPath.length > 0) {
                    map.fitBounds(L.polyline(coordinatesPath).getBounds(), { padding: [50, 50] });
                }
                addLog(`🏁 Demo route simulation complete.`, "success");
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

