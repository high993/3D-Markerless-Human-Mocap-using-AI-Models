import { useEffect, useState } from 'react';
import io from 'socket.io-client';
import ThreeDChart from './3DChart';

export default function App() {
  const [currentFrameData, setCurrentFrameData] = useState<number[][] | null>(null); // Store the current frame data
  const [selectedCamera, setSelectedCamera] = useState<number>(2); // Store selected camera index

  useEffect(() => {
    // Initialize Socket.IO connection within useEffect
    const socket = io('http://localhost:3001');

    // Establish Socket.IO connection and log success
    socket.on('connect', () => {
      console.log('Connected to server via Socket.IO');
    });

    // Listen for '3d-points' events from the backend
    socket.on('3d-points', (frameData: number[][]) => {
      console.log('Received frame data:', frameData);
      setCurrentFrameData(frameData);
    });

    // Cleanup on component unmount
    return () => {
      socket.disconnect(); // Clean up when component unmounts
      console.log('Socket.IO connection closed');
    };
  }, []); // Empty dependency array to ensure this effect runs only on mount

  const handleButtonClick = async (endpoint: string) => {
    try {
      const response = await fetch(`http://localhost:3001/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();
      console.log(data.message); // Log the backend response
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handlePlaybackClick = () => {
    handleButtonClick('run-playback');
  };

  // Function to handle camera selection
  const handleCameraSelection = (cameraIndex: number) => {
    setSelectedCamera(cameraIndex); // Update the selected camera state
    // Emit the camera index to the backend via Socket.IO
    const socket = io('http://localhost:3001');
    socket.emit('change_camera_index', { camera_index: cameraIndex });
    console.log(`Camera index ${cameraIndex} selected and emitted`);
  };

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Top control bar */}
      <div
        style={{
          height: '50px',
          width: '100%',
          backgroundColor: 'grey',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          padding: '0 10px',
        }}
      >
        {/* Camera selection buttons */}
        <button style={{ marginRight: '10px' }} onClick={() => handleCameraSelection(0)}>Select Camera 0</button>
        <button style={{ marginRight: '10px' }} onClick={() => handleCameraSelection(1)}>Select Camera 1</button>
        <button style={{ marginRight: '10px' }} onClick={() => handleCameraSelection(2)}>Select Camera 2</button>
        <button style={{ marginRight: '10px' }} onClick={() => handleCameraSelection(3)}>Select Camera 3</button>
        <span style={{ marginLeft: '20px' }}>Currently Selected Camera: {selectedCamera}</span>
        
      </div>

      <div style={{ display: 'flex', flexGrow: 1 }}>
        {/* Left control bar */}
        <div
          style={{
            width: '150px',
            backgroundColor: 'grey',
            color: 'white',
            display: 'flex',
            flexDirection: 'column',
            padding: '10px',
          }}
        >
          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-gather-intrinsic')}
          >
            Gather Intrinsic Data for Specific Cam
          </button>
          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-calibrate-camera')}
          >
            Calculate Intrinsic for Specific Cam
          </button>
          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-gatherfloor')}
          >
            Gather Floor Data
          </button>
          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-extrinsic-calibration')}
          >
            Extrinsic Calibration
          </button>

          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-capture-data')}
          >
            Capture Data
          </button>

          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-pose')}
          >
            Get 2D Data
          </button>
          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-3D-Triangulation')}
          >
            Get 3D Data
          </button>
          <button
            style={{ marginBottom: '10px' }}
            onClick={handlePlaybackClick}
          >
            PlayBack in 3D
          </button>

          <button
            style={{ marginBottom: '10px' }}
            onClick={() => handleButtonClick('run-add-data')}
          >
            Add Data to Neural Net
          </button>
        </div>

        {/* 3D Chart */}
        <div style={{ flexGrow: 1 }}>
          <ThreeDChart frameData={currentFrameData} />
        </div>
      </div>
    </div>
  );
}
