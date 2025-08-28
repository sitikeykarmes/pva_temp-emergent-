from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import cv2
import asyncio
import time
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from parking_detection import ParkingDetectionSystem

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'parking_db')]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize parking detection system
parking_system = ParkingDetectionSystem()

# Create videos directory if it doesn't exist
videos_dir = ROOT_DIR / "videos"
videos_dir.mkdir(exist_ok=True)

# Available videos mapping
AVAILABLE_VIDEOS = {
    "AB-1 Parking": "videos/ab1.mp4",
    "AB-3 Parking": "videos/ab3.mp4", 
    "Ab-3 Front": "videos/ab3_front.mp4",
    "GymKhana": "videos/gymkhana_1.mp4",
    "AB-1 Front": "videos/ab1_front.mp4",
    "Aavin": "videos/aavin.mp4",
    "Vmart": "videos/Vmart_1.mp4",
    "Sigma Block": "videos/sigmablock.mp4"
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class ViolationLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vehicle_id: int
    location: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration: float
    violation_type: str = "no_parking_zone"

# Basic routes
@api_router.get("/")
async def root():
    return {"message": "Parking Detection System API"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Video-related routes
@api_router.get("/videos")
async def get_available_videos():
    """Get list of available videos"""
    available = {}
    for name, path in AVAILABLE_VIDEOS.items():
        full_path = ROOT_DIR / path
        if full_path.exists():
            available[name] = path
        else:
            # For demo purposes, we'll still return the path even if file doesn't exist
            available[name] = path
    
    return {
        "videos": available,
        "total": len(available)
    }

@api_router.get("/video/{video_name}")
async def get_video_file(video_name: str):
    """Serve video files"""
    if video_name not in AVAILABLE_VIDEOS:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = ROOT_DIR / AVAILABLE_VIDEOS[video_name]
    
    # For demo purposes, create a placeholder if video doesn't exist
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    return FileResponse(video_path)

@api_router.post("/process-frame")
async def process_frame_endpoint(request: dict):
    """Process a single frame for parking detection"""
    try:
        # This would typically receive base64 encoded frame data
        # For now, we'll return mock data
        
        frame_data = request.get('frame_data')
        video_name = request.get('video_name', 'unknown')
        
        # Mock processing result
        result = {
            'prediction': 'No Parking Zone',
            'vehicles': [
                {
                    'id': 1,
                    'bbox': [100, 100, 200, 200],
                    'class': 'car',
                    'confidence': 0.95,
                    'duration': 3.5,
                    'status': 'normal'
                }
            ],
            'alerts': [],
            'timestamp': time.time(),
            'processing_time': 0.05
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/reset-alerts")
async def reset_alerts():
    """Reset all parking violation alerts"""
    try:
        parking_system.reset_alerts()
        
        # Broadcast reset to all connected clients
        await manager.broadcast({
            'type': 'alerts_reset',
            'timestamp': time.time()
        })
        
        return {"message": "Alerts reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/violations", response_model=ViolationLog)
async def log_violation(violation: ViolationLog):
    """Log a parking violation"""
    try:
        violation_dict = violation.dict()
        _ = await db.violations.insert_one(violation_dict)
        
        # Broadcast violation to all connected clients
        await manager.broadcast({
            'type': 'new_violation',
            'data': violation_dict
        })
        
        return violation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/violations", response_model=List[ViolationLog])
async def get_violations():
    """Get all parking violations"""
    try:
        violations = await db.violations.find().sort("timestamp", -1).to_list(100)
        return [ViolationLog(**violation) for violation in violations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Include the router in the main app
app.include_router(api_router)

# Serve static files (videos)
app.mount("/static", StaticFiles(directory=str(videos_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)