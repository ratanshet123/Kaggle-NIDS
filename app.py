from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pydantic import BaseModel

# Load model and scaler
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI
app = FastAPI()

# Define categorical mappings
protocol_map = {"tcp": 0, "udp": 1, "icmp": 2}
service_map = {"http": 0, "ftp": 1, "ssh": 2, "smtp": 3, "dns": 4, "private": 5}  

flag_map = {"SF": 0, "S0": 1, "REJ": 2, "RSTO": 3, "SH": 4}  

# Define request body
class IntrusionData(BaseModel):
    duration: int
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

# Graphing endpoint
@app.get("/graphs")
def get_graphs():
    try:
        return {"message": "Graphs generated successfully!", "graph_url": "confusion_matrix.png"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graphs: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Root endpoint


@app.get("/")
def home():
    return {"message": "Intrusion Detection System API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: IntrusionData):
    try:
        # Convert categorical values to numbers
        protocol = protocol_map.get(data.protocol_type, -1)
        service = service_map.get(data.service, -1)
        flag = flag_map.get(data.flag, -1)

        # If a categorical value is not found in the mapping
        if -1 in [protocol, service, flag]:
            raise ValueError("Invalid categorical value: protocol, service, or flag not recognized.")

        # Convert input data to a list of numerical features
        features = [
            data.duration, protocol, service, flag, data.src_bytes, data.dst_bytes,
            data.land, data.wrong_fragment, data.urgent, data.hot, data.num_failed_logins,
            data.logged_in, data.num_compromised, data.root_shell, data.su_attempted,
            data.num_root, data.num_file_creations, data.num_shells, data.num_access_files,
            data.num_outbound_cmds, data.is_host_login, data.is_guest_login, data.count,
            data.srv_count, data.serror_rate, data.srv_serror_rate, data.rerror_rate,
            data.srv_rerror_rate, data.same_srv_rate, data.diff_srv_rate, data.srv_diff_host_rate,
            data.dst_host_count, data.dst_host_srv_count, data.dst_host_same_srv_rate,
            data.dst_host_diff_srv_rate, data.dst_host_same_src_port_rate, data.dst_host_srv_diff_host_rate,
            data.dst_host_serror_rate, data.dst_host_srv_serror_rate, data.dst_host_rerror_rate,
            data.dst_host_srv_rerror_rate
        ]

        # Check if the input matches expected feature count
        expected_features = scaler.n_features_in_
        if len(features) != expected_features:
            raise ValueError(f"Expected {expected_features} features, but got {len(features)}.")

        # Convert to numpy array and reshape
        input_data = np.array(features, dtype=float).reshape(1, -1)

        # Normalize input
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        return {"intrusion_detected": bool(prediction[0])}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")