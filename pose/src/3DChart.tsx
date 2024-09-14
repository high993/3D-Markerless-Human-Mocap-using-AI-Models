import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// Component to render individual grid points
function GridPoints({ data }: { data: number[][] | null }) {
  if (!data) return null; // No data, render nothing

  return (
    <>
      {data.map((point, index) => (
        point.length === 3 && (
          // Convert from mm to cm and adjust axis: swapping Y and Z
          <mesh key={index} position={[point[0] / 10, point[2] / 10, point[1] / 10] as [number, number, number]}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial color={'skyblue'} />
          </mesh>
        )
      ))}
    </>
  );
}

function DrawLinesBetweenPoints({ points, connections }: { points: number[][] | null, connections: [number, number][] }) {
  if (!points) return null;

  return (
    <>
      {connections.map(([start, end], index) => {
        const startPoint = points[start];
        const endPoint = points[end];

        // Ensure both start and end points exist and are valid (not null, undefined, or containing invalid values)
        if (
          startPoint && endPoint &&
          startPoint.length === 3 && endPoint.length === 3 &&
          startPoint.every(coord => coord !== null && coord !== undefined) &&
          endPoint.every(coord => coord !== null && coord !== undefined)
        ) {
          const startPos = new THREE.Vector3(startPoint[0] / 10, startPoint[2] / 10, startPoint[1] / 10);
          const endPos = new THREE.Vector3(endPoint[0] / 10, endPoint[2] / 10, endPoint[1] / 10);

          const pointsArray = [startPos, endPos];
          const lineGeometry = new THREE.BufferGeometry().setFromPoints(pointsArray);

          return (
            <line key={index}>
              <bufferGeometry attach="geometry" {...lineGeometry} />
              <lineBasicMaterial attach="material" color={'red'} linewidth={2} />
            </line>
          );
        }
        return null; // Skip drawing the line if points are invalid or missing
      })}
    </>
  );
}

// Component for the ground plane
function Plane() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[300, 300]} />
      <meshStandardMaterial color={'lightgray'} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Component for the origin marker
function OriginMarker() {
  return (
    <mesh position={[0, 0, 0]}>
      <sphereGeometry args={[0.3, 64, 64]} />
      <meshStandardMaterial color={'red'} />
    </mesh>
  );
}

// Grid helper for the scene
function GridHelper() {
  const gridSize = 10;
  const gridDivisions = 10;

  return <gridHelper args={[gridSize, gridDivisions]} position={[0, 0, 0]} />;
}

// Main 3D chart component
export default function ThreeDChart({ frameData }: { frameData: number[][] | null }) {
  // Define the connection to draw a line between point 12 and point 14
  const connections: [number, number][] = [
    [12, 14],[14,16],[16,22],[16,20],[18,20],[16,18],[12,24],[24,26],[26,28],[28,30],[30,32],[32,28],[12,11],[24,23],[11,13],[13,15],[15,21],[15,19],[19,17],[15,17],[11,23],[23,25],[25,27],[27,31],[31,29],[29,27]
     // Line from point 12 to point 14
    // You can add more connections here if needed
  ];

  return (
    <Canvas>
      <ambientLight />
      <pointLight position={[10, 10, 10]} />
      <Plane />
      <GridHelper />
      <OriginMarker />
      <GridPoints data={frameData} />
      <DrawLinesBetweenPoints points={frameData} connections={connections} />
      <OrbitControls />
    </Canvas>
  );
}
