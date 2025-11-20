import React, { useState, useEffect, useRef } from 'react';
import { Variable, RefreshCw, PlayCircle, Info } from 'lucide-react';

const PhysicsEngineDemo: React.FC = () => {
  const [k, setK] = useState(1.0); // Einstein Radius approx
  const [q, setQ] = useState(0.8); // Axis Ratio
  const [phi, setPhi] = useState(45); // Orientation angle
  const [substructure, setSubstructure] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Ray-Tracing Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Image Data containers
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Source Galaxy Parameters (Gaussian Blob)
    const sourceX = 0; 
    const sourceY = 0;
    const sourceSigma = 15;

    // Pre-calculate rotation sin/cos for efficiency
    const rad = (phi * Math.PI) / 180;
    const cosPhi = Math.cos(rad);
    const sinPhi = Math.sin(rad);

    // Render Loop: Inverse Ray-Tracing
    // For each pixel in the IMAGE PLANE (theta), find where it came from in SOURCE PLANE (beta)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        // 1. Normalize coordinates centered at 0
        const dx = x - cx;
        const dy = y - cy;

        // 2. Rotate coordinates to align with lens axis
        const rx = dx * cosPhi + dy * sinPhi;
        const ry = -dx * sinPhi + dy * cosPhi;

        // 3. Calculate Deflection (Simplified SIE approximation for visual demo)
        // Distance in the elliptical potential
        // q is axis ratio b/a. 
        const dist = Math.sqrt(q * q * rx * rx + ry * ry) + 0.001; // Avoid div/0

        // Deflection magnitude (alpha) proportional to Einstein Radius (k) scaled roughly to pixels
        // k=1.0 corresponds to roughly 50px radius
        const alphaMag = k * 50; 

        // Deflection angle components
        const alphax = alphaMag * (Math.sqrt(q) * rx / dist);
        const alphay = alphaMag * (Math.sqrt(q) * ry / dist);

        // 4. Substructure Perturbations (Simulating CDM clumps)
        let perturbX = 0;
        let perturbY = 0;
        if (substructure) {
            // Simple deterministic noise based on position
            perturbX = Math.sin(dx * 0.1) * 4 * Math.cos(dy * 0.1);
            perturbY = Math.cos(dx * 0.1) * 4 * Math.sin(dy * 0.1);
        }

        // 5. Lens Equation: Beta = Theta - Alpha
        // Rotate deflection back to original frame
        const defX = alphax * cosPhi - alphay * sinPhi;
        const defY = alphax * sinPhi + alphay * cosPhi;

        const betaX = dx - (defX + perturbX);
        const betaY = dy - (defY + perturbY);

        // 6. Sample Source Intensity (Gaussian)
        // Source is at (sourceX, sourceY) in the source plane (which corresponds to center of screen if unlensed)
        const sx = betaX - sourceX;
        const sy = betaY - sourceY;
        const rSq = sx*sx + sy*sy;
        
        // Intensity 0-255
        const intensity = Math.exp(-rSq / (2 * sourceSigma * sourceSigma)) * 255;

        // 7. Write to pixel data
        const index = (y * width + x) * 4;
        
        // Colorize based on intensity (Golden/White galaxy color)
        data[index] = intensity;     // R
        data[index + 1] = intensity * 0.8; // G
        data[index + 2] = intensity * 0.6; // B
        data[index + 3] = 255;       // Alpha
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Draw Labels overlay
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();

  }, [k, q, phi, substructure]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Controls */}
      <div className="glass-panel p-6 rounded-xl flex flex-col h-full">
        <div className="mb-6">
          <h3 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
            <Variable className="text-cyan-400" />
            Interactive Lens Simulator
          </h3>
          <p className="text-gray-300 text-sm">
            Real-time ray-tracing of the <span className="font-mono text-cyan-300">Lens Equation β = θ - α(θ)</span>. 
            Observe how mass distribution parameters distort the background source galaxy into Einstein Rings or arcs.
          </p>
        </div>

        <div className="space-y-8 flex-1">
          {/* Slider K */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <label className="text-gray-300 font-medium">Einstein Radius (θ_E)</label>
              <span className="font-mono text-cyan-400 bg-cyan-950/50 px-2 rounded">{k.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0.0" max="2.5" step="0.1" 
              value={k} onChange={(e) => setK(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <p className="text-xs text-gray-500 mt-1">Proportional to the total mass of the lens.</p>
          </div>

          {/* Slider Q */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <label className="text-gray-300 font-medium">Axis Ratio (q)</label>
              <span className="font-mono text-cyan-400 bg-cyan-950/50 px-2 rounded">{q.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0.1" max="1.0" step="0.05" 
              value={q} onChange={(e) => setQ(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            <p className="text-xs text-gray-500 mt-1">From 1.0 (Spherical) to 0.1 (Highly Elliptical).</p>
          </div>

          {/* Slider Phi */}
          <div>
            <div className="flex justify-between text-sm mb-2">
              <label className="text-gray-300 font-medium">Orientation (φ)</label>
              <span className="font-mono text-cyan-400 bg-cyan-950/50 px-2 rounded">{phi}°</span>
            </div>
            <input 
              type="range" min="0" max="180" step="1" 
              value={phi} onChange={(e) => setPhi(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
          </div>

          {/* Substructure Toggle */}
          <div className="pt-4 border-t border-gray-800">
             <div className="flex items-center justify-between">
                <label className="text-gray-300 font-medium flex items-center gap-2">
                   <Info className="w-4 h-4 text-purple-400" />
                   Simulate Substructure (CDM)
                </label>
                <button 
                  onClick={() => setSubstructure(!substructure)}
                  className={`w-12 h-6 rounded-full transition-colors relative ${substructure ? 'bg-purple-600' : 'bg-gray-700'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${substructure ? 'left-7' : 'left-1'}`} />
                </button>
             </div>
             <p className="text-xs text-gray-500 mt-2">
               Adds small-scale gravitational perturbations characteristic of Cold Dark Matter particle halos.
             </p>
          </div>
        </div>
      </div>

      {/* Visualizer */}
      <div className="glass-panel p-1 rounded-xl flex items-center justify-center bg-black relative aspect-square border border-gray-700 shadow-2xl">
        <canvas 
          ref={canvasRef}
          width={400}
          height={400}
          className="w-full h-full rounded-lg"
        />
        <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded border border-gray-700 text-xs text-gray-300 font-mono">
          Real-time Render: 400x400
        </div>
        <div className="absolute bottom-4 right-4 flex gap-2">
             <button onClick={() => {setK(1.0); setQ(0.8); setPhi(45); setSubstructure(false)}} className="p-2 bg-gray-800 hover:bg-gray-700 rounded-full text-white border border-gray-600" title="Reset">
                <RefreshCw className="w-4 h-4" />
             </button>
        </div>
      </div>
    </div>
  );
};

export default PhysicsEngineDemo;