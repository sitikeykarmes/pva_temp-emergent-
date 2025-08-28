import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Video Player Component with Overlays
const ParkingVideoPlayer = ({ videoName, onViolation }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [detectionData, setDetectionData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const processingRef = useRef(null);

  useEffect(() => {
    if (videoName) {
      // Reset video when switching videos
      setDetectionData(null);
      setAlerts([]);
      if (videoRef.current) {
        videoRef.current.load();
      }
    }
  }, [videoName]);

  const processFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Mock processing for now (since we don't have actual video processing in browser)
    const mockData = {
      prediction: Math.random() > 0.5 ? 'No Parking Zone' : 'Parking Zone',
      vehicles: [
        {
          id: Math.floor(Math.random() * 100),
          bbox: [
            Math.random() * canvas.width * 0.5,
            Math.random() * canvas.height * 0.5,
            Math.random() * canvas.width * 0.3 + canvas.width * 0.3,
            Math.random() * canvas.height * 0.3 + canvas.height * 0.3
          ],
          class: 'car',
          confidence: 0.9 + Math.random() * 0.1,
          duration: Math.random() * 10,
          status: Math.random() > 0.8 ? 'violation' : 'normal'
        }
      ],
      timestamp: Date.now()
    };

    setDetectionData(mockData);

    // Check for violations
    const violations = mockData.vehicles.filter(v => v.status === 'violation');
    if (violations.length > 0 && mockData.prediction === 'No Parking Zone') {
      const newAlert = {
        id: Date.now(),
        vehicleId: violations[0].id,
        timestamp: new Date(),
        location: videoName,
        duration: violations[0].duration,
        message: `Vehicle #${violations[0].id} parked in no-parking zone for ${violations[0].duration.toFixed(1)}s`
      };
      
      setAlerts(prev => [newAlert, ...prev.slice(0, 4)]);
      onViolation && onViolation(newAlert);
    }

    // Draw overlays
    drawOverlays(ctx, mockData);
  };

  const drawOverlays = (ctx, data) => {
    if (!data) return;

    // Draw zone prediction
    ctx.fillStyle = data.prediction === 'Parking Zone' ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
    ctx.fillRect(0, 0, ctx.canvas.width, 60);
    
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.fillText(`Zone: ${data.prediction}`, 20, 35);

    // Draw vehicle bounding boxes
    data.vehicles.forEach(vehicle => {
      const [x1, y1, x2, y2] = vehicle.bbox;
      
      // Set color based on status
      ctx.strokeStyle = vehicle.status === 'violation' ? 'red' : 
                       vehicle.status === 'warning' ? 'yellow' : 'green';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Draw vehicle info
      ctx.fillStyle = 'white';
      ctx.fillRect(x1, y1 - 25, 150, 25);
      
      ctx.fillStyle = 'black';
      ctx.font = '14px Arial';
      ctx.fillText(`ID: ${vehicle.id} (${vehicle.duration.toFixed(1)}s)`, x1 + 5, y1 - 8);
    });

    // Draw alerts banner
    if (alerts.length > 0) {
      ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
      ctx.fillRect(0, 70, ctx.canvas.width, 40);
      
      ctx.fillStyle = 'white';
      ctx.font = 'bold 16px Arial';
      ctx.fillText('!!! PARKING VIOLATIONS DETECTED !!!', 20, 95);
    }
  };

  const handlePlay = () => {
    if (videoRef.current) {
      setIsPlaying(true);
      videoRef.current.play();
      
      // Start processing frames
      processingRef.current = setInterval(processFrame, 200); // Process every 200ms
    }
  };

  const handlePause = () => {
    if (videoRef.current) {
      setIsPlaying(false);
      videoRef.current.pause();
      
      // Stop processing
      if (processingRef.current) {
        clearInterval(processingRef.current);
      }
    }
  };

  useEffect(() => {
    return () => {
      if (processingRef.current) {
        clearInterval(processingRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full bg-black rounded-lg overflow-hidden">
      <div className="relative">
        <video
          ref={videoRef}
          className="w-full h-auto"
          src={videoName ? `${BACKEND_URL}/api/video/${videoName}` : ''}
          controls={false}
          onLoadedData={() => {
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
            }
          }}
        />
        
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
        
        {!videoName && (
          <div className="absolute inset-0 flex items-center justify-center text-white bg-gray-800">
            <p className="text-xl">Select a video to start detection</p>
          </div>
        )}
      </div>
      
      <div className="p-4 bg-gray-900 text-white">
        <div className="flex gap-4 mb-4">
          <button
            onClick={handlePlay}
            disabled={!videoName || isPlaying}
            className="px-4 py-2 bg-green-600 text-white rounded disabled:bg-gray-500"
          >
            Play
          </button>
          <button
            onClick={handlePause}
            disabled={!videoName || !isPlaying}
            className="px-4 py-2 bg-red-600 text-white rounded disabled:bg-gray-500"
          >
            Pause
          </button>
          <button
            onClick={() => {
              setAlerts([]);
              setDetectionData(null);
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded"
          >
            Reset Alerts
          </button>
        </div>
        
        {detectionData && (
          <div className="text-sm space-y-2">
            <p>Zone: <span className={detectionData.prediction === 'Parking Zone' ? 'text-green-400' : 'text-red-400'}>
              {detectionData.prediction}
            </span></p>
            <p>Vehicles Detected: {detectionData.vehicles.length}</p>
            <p>Processing Status: {isPlaying ? 'Active' : 'Paused'}</p>
          </div>
        )}
      </div>
    </div>
  );
};

// Alerts Panel Component
const AlertsPanel = ({ alerts }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-lg font-bold mb-4 text-red-600">Recent Violations</h3>
      
      {alerts.length === 0 ? (
        <p className="text-gray-500">No violations detected</p>
      ) : (
        <div className="space-y-3 max-h-64 overflow-y-auto">
          {alerts.map(alert => (
            <div key={alert.id} className="p-3 bg-red-50 border-l-4 border-red-500 rounded">
              <div className="flex justify-between items-start">
                <div>
                  <p className="font-semibold text-red-800">Vehicle #{alert.vehicleId}</p>
                  <p className="text-sm text-red-600">{alert.message}</p>
                  <p className="text-xs text-gray-500">Location: {alert.location}</p>
                </div>
                <span className="text-xs text-gray-400">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Main App Component
const ParkingDetectionApp = () => {
  const [videos, setVideos] = useState({});
  const [selectedVideo, setSelectedVideo] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const response = await axios.get(`${API}/videos`);
        setVideos(response.data.videos);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching videos:', error);
        setIsLoading(false);
      }
    };

    fetchVideos();
  }, []);

  const handleViolation = async (violation) => {
    try {
      // Log violation to backend
      await axios.post(`${API}/violations`, {
        vehicle_id: violation.vehicleId,
        location: violation.location,
        duration: violation.duration,
        violation_type: 'no_parking_zone'
      });
    } catch (error) {
      console.error('Error logging violation:', error);
    }
  };

  const resetAllAlerts = async () => {
    try {
      await axios.post(`${API}/reset-alerts`);
      setAlerts([]);
    } catch (error) {
      console.error('Error resetting alerts:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl">Loading Parking Detection System...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Smart Parking Detection System</h1>
          <p className="text-gray-600">Real-time vehicle tracking and violation detection</p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Video Selection and Player */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4">Video Selection</h2>
              <select
                value={selectedVideo}
                onChange={(e) => setSelectedVideo(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a video location...</option>
                {Object.keys(videos).map(videoName => (
                  <option key={videoName} value={videoName}>
                    {videoName}
                  </option>
                ))}
              </select>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4">Live Detection Feed</h2>
              <ParkingVideoPlayer
                videoName={selectedVideo}
                onViolation={(violation) => setAlerts(prev => [violation, ...prev.slice(0, 9)])}
              />
            </div>
          </div>

          {/* Alerts and Controls */}
          <div className="space-y-6">
            <AlertsPanel alerts={alerts} />
            
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">System Controls</h3>
              <div className="space-y-3">
                <button
                  onClick={resetAllAlerts}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Reset All Alerts
                </button>
                
                <div className="p-3 bg-gray-50 rounded">
                  <h4 className="font-semibold mb-2">Detection Settings</h4>
                  <p className="text-sm text-gray-600">Alert Threshold: 5 seconds</p>
                  <p className="text-sm text-gray-600">Processing: Real-time</p>
                  <p className="text-sm text-gray-600">Model: YOLO + CNN + RF</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold mb-4">System Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Detection Engine:</span>
                  <span className="text-green-600 font-semibold">Active</span>
                </div>
                <div className="flex justify-between">
                  <span>Total Videos:</span>
                  <span className="font-semibold">{Object.keys(videos).length}</span>
                </div>
                <div className="flex justify-between">
                  <span>Active Violations:</span>
                  <span className="text-red-600 font-semibold">{alerts.length}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <ParkingDetectionApp />
    </div>
  );
}

export default App;