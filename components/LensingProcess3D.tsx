import React, { useRef, useEffect } from 'react';
import { Play, Info } from 'lucide-react';

const LensingProcess3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = canvas.offsetWidth;
    let height = canvas.offsetHeight;

    // Resize handler
    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };
    window.addEventListener('resize', handleResize);
    handleResize();

    // --- Constants & Config ---
    // World Coordinates: 
    // Z axis: The long axis of travel (Source -> Earth)
    // X axis: Horizontal width of the volume
    // Y axis: Vertical height of the volume
    
    const Z_START = -400;
    const Z_LENS = 0;
    const Z_END = 400;
    
    const COLOR_SOURCE = '#60a5fa'; // Blue-400
    const COLOR_LENS = '#fbbf24';   // Amber-400
    const COLOR_RAY = '#ccfbf1';    // Teal-100
    const COLOR_OBSERVER = '#22c55e'; // Green-500

    // --- Helper: Isometric Projection ---
    const project = (x: number, y: number, z: number) => {
      // Scale relative to canvas size to keep it responsive
      const s = Math.min(width, height) / 1000; 
      
      // Diagonal Projection Matrix simulation
      // We want Z to run diagonally from Top-Left to Bottom-Right
      const px = width * 0.5 + (z * 0.5 + x * 0.8) * s; 
      const py = height * 0.5 + (z * 0.25 - x * 0.3 + y) * s; 
      
      // Simple depth scaling for point size
      const scale = 0.5 + ((z - Z_START) / (Z_END - Z_START)) * 0.5;
      
      return { x: px, y: py, scale };
    };

    // --- Classes ---
    class Photon {
      x: number;
      y: number;
      z: number;
      
      targetR: number; // Radius at lens plane
      targetTheta: number; // Angle at lens plane
      
      progress: number;
      speed: number;
      
      constructor() {
        this.reset();
        // Randomize initial progress to create a stream
        this.progress = Math.random(); 
        this.speed = 0.005 + Math.random() * 0.002;
        this.targetR = 100;
        this.targetTheta = 0;
        this.x = 0;
        this.y = 0;
        this.z = 0;
      }

      reset() {
        this.progress = 0;
        // Target a point on the Einstein Ring (Lens Plane)
        // The "Ring" radius is roughly 120 units in world space
        this.targetR = 100 + (Math.random() - 0.5) * 20;
        this.targetTheta = Math.random() * Math.PI * 2;
      }

      update() {
        this.progress += this.speed;
        if (this.progress >= 1) {
          this.reset();
        }

        // Path Logic:
        // 0.0 -> 0.5: Source to Lens
        // 0.5 -> 1.0: Lens to Observer
        
        if (this.progress < 0.5) {
          // Phase 1: Source -> Lens
          const t = this.progress * 2; // 0 to 1
          
          // Start: (0,0, Z_START)
          // End: (rx, ry, Z_LENS)
          const tx = Math.cos(this.targetTheta) * this.targetR;
          const ty = Math.sin(this.targetTheta) * this.targetR * 0.8; // Flattened slightly
          
          this.x = tx * t;
          this.y = ty * t;
          this.z = Z_START + (Z_LENS - Z_START) * t;
          
        } else {
          // Phase 2: Lens -> Observer
          const t = (this.progress - 0.5) * 2; // 0 to 1
          
          // Start: (rx, ry, Z_LENS)
          // End: (0,0, Z_END) - The Observer
          const sx = Math.cos(this.targetTheta) * this.targetR;
          const sy = Math.sin(this.targetTheta) * this.targetR * 0.8;
          
          this.x = sx * (1 - t);
          this.y = sy * (1 - t);
          this.z = Z_LENS + (Z_END - Z_LENS) * t;
        }
      }

      draw(ctx: CanvasRenderingContext2D) {
        const p = project(this.x, this.y, this.z);
        
        // Trail (simple previous position calculation)
        // To make it fast, we just draw a line from slightly back in time
        const tailZ = this.z - 40;
        // Approximate tail position linearly (good enough for high speed visualization)
        let tailX = this.x; 
        let tailY = this.y;
        
        // Adjust tail based on phase for curvature
        if (Math.abs(this.z - Z_LENS) < 50) {
            // Near lens, don't draw long straight tails to avoid visual glitching at the bend
            // just draw dot
        } else {
             // Simple projection towards center for tail if converging/diverging
             // This is purely cosmetic
             const factor = 0.9; // shrinking towards origin
             if (this.z > Z_LENS) { tailX = this.x / factor; tailY = this.y / factor; } // Converging
             else { tailX = this.x * factor; tailY = this.y * factor; } // Diverging
        }

        // Draw Photon Head
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2 * p.scale, 0, Math.PI * 2);
        ctx.fillStyle = COLOR_RAY;
        ctx.fill();
      }
    }

    const photons: Photon[] = Array.from({ length: 80 }, () => new Photon());

    // --- Render Loop ---
    const render = () => {
      // Clear
      ctx.fillStyle = '#030712';
      ctx.fillRect(0, 0, width, height);
      
      // 1. Draw Volume Box (Wireframe)
      const boxW = 200;
      const boxH = 150;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      
      const corners = [
        [-boxW, -boxH], [boxW, -boxH], [boxW, boxH], [-boxW, boxH]
      ];
      
      // Draw Start Plane (Source)
      const startPoints = corners.map(c => project(c[0], c[1], Z_START));
      ctx.beginPath();
      startPoints.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.stroke();

      // Draw End Plane (Observer)
      const endPoints = corners.map(c => project(c[0], c[1], Z_END));
      ctx.beginPath();
      endPoints.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.stroke();

      // Connect Corners
      for (let i = 0; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(startPoints[i].x, startPoints[i].y);
        ctx.lineTo(endPoints[i].x, endPoints[i].y);
        ctx.stroke();
      }
      
      // Draw Middle Plane (Lens) - Faint
      const midPoints = corners.map(c => project(c[0], c[1], Z_LENS));
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      midPoints.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);


      // 2. Draw Source Galaxy (Top Left / Z_START)
      const pSource = project(0, 0, Z_START);
      // Glow
      const gradS = ctx.createRadialGradient(pSource.x, pSource.y, 0, pSource.x, pSource.y, 40 * pSource.scale);
      gradS.addColorStop(0, 'rgba(96, 165, 250, 0.8)');
      gradS.addColorStop(1, 'rgba(96, 165, 250, 0)');
      ctx.fillStyle = gradS;
      ctx.beginPath();
      ctx.arc(pSource.x, pSource.y, 40 * pSource.scale, 0, Math.PI*2);
      ctx.fill();
      // Core
      ctx.fillStyle = COLOR_SOURCE;
      ctx.beginPath();
      ctx.ellipse(pSource.x, pSource.y, 20 * pSource.scale, 10 * pSource.scale, -0.2, 0, Math.PI*2);
      ctx.fill();

      // 3. Draw Lens (Center / Z_LENS)
      const pLens = project(0, 0, Z_LENS);
      // DM Halo (Fuzzy)
      const gradL = ctx.createRadialGradient(pLens.x, pLens.y, 0, pLens.x, pLens.y, 80 * pLens.scale);
      gradL.addColorStop(0, 'rgba(251, 191, 36, 0.15)'); // Amber transparent
      gradL.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = gradL;
      ctx.beginPath();
      ctx.arc(pLens.x, pLens.y, 80 * pLens.scale, 0, Math.PI*2);
      ctx.fill();
      // Galaxies in Cluster
      ctx.fillStyle = '#fcd34d'; // Amber-300
      [ [0,0], [20,10], [-15, 20], [10, -25], [-20, -10] ].forEach(offset => {
         const gx = pLens.x + offset[0] * pLens.scale;
         const gy = pLens.y + offset[1] * pLens.scale;
         ctx.beginPath();
         ctx.ellipse(gx, gy, 6 * pLens.scale, 4 * pLens.scale, Math.random(), 0, Math.PI*2);
         ctx.fill();
      });

      // 4. Draw Rays / Photons
      photons.forEach(photon => {
        photon.update();
        photon.draw(ctx);
      });

      // 5. Draw Observer (Bottom Right / Z_END)
      const pObs = project(0, 0, Z_END);
      // Earth Icon
      ctx.fillStyle = COLOR_OBSERVER;
      ctx.beginPath();
      ctx.arc(pObs.x, pObs.y, 6 * pObs.scale, 0, Math.PI*2);
      ctx.fill();
      
      // Telescope beam lines
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.3)';
      ctx.beginPath();
      ctx.moveTo(pObs.x, pObs.y);
      ctx.lineTo(pObs.x - 30, pObs.y - 50); // Beam pointing up-left
      ctx.stroke();


      // 6. LABELS & OVERLAYS (Callouts)
      ctx.font = "bold 12px JetBrains Mono";
      ctx.textAlign = "center";

      // Label 1: Source
      ctx.fillStyle = COLOR_SOURCE;
      // Adjusted: Moved up by 60 units (scaled) to ensure it clears the galaxy
      ctx.fillText("1. DISTANT SOURCE", pSource.x, pSource.y - 60 * pSource.scale);

      // Label 2: Lens
      ctx.fillStyle = "#fbbf24";
      // Adjusted: Moved BELOW the lens cluster to avoid overlap with Source label/rays
      ctx.fillText("2. DARK MATTER LENS", pLens.x, pLens.y + 110 * pLens.scale);

      // Label 3: Observer
      ctx.fillStyle = COLOR_OBSERVER;
      ctx.fillText("3. EARTH", pObs.x, pObs.y + 35 * pObs.scale);

      // 7. "OBSERVED IMAGE" Callout (Inset)
      // Draw a line from Earth to the callout box
      const calloutX = width - 160;
      const calloutY = 80;
      const calloutSize = 120;
      
      // Connecting Line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(pObs.x, pObs.y);
      ctx.lineTo(calloutX + calloutSize/2, calloutY + calloutSize);
      ctx.stroke();
      ctx.setLineDash([]);

      // Box Background
      ctx.fillStyle = '#000';
      ctx.strokeStyle = '#374151'; // Gray-700
      ctx.lineWidth = 1;
      ctx.fillRect(calloutX, calloutY, calloutSize, calloutSize);
      ctx.strokeRect(calloutX, calloutY, calloutSize, calloutSize);

      // Draw Einstein Ring inside box
      const cx = calloutX + calloutSize/2;
      const cy = calloutY + calloutSize/2;
      const r = 35;
      
      // Ring
      ctx.strokeStyle = COLOR_SOURCE;
      ctx.lineWidth = 3;
      ctx.shadowColor = COLOR_SOURCE;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI*2);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1;
      
      // Central bright spot (Lens galaxy)
      ctx.fillStyle = '#fcd34d';
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI*2);
      ctx.fill();

      // Label
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.font = "12px JetBrains Mono"; // Reset font for this label if needed, or keep bold
      ctx.fillText("4. GALAXY OBSERVATION", calloutX, calloutY - 10);


      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className="w-full h-[500px] bg-gray-950 rounded-xl overflow-hidden relative border border-gray-800 shadow-2xl">
        <canvas ref={canvasRef} className="w-full h-full block" />
        
        {/* Legend/Info Overlay */}
        <div className="absolute bottom-4 left-4 p-4 bg-black/60 backdrop-blur-md rounded-lg border border-gray-700 max-w-md pointer-events-none">
            <div className="flex items-center gap-2 mb-2">
                <Play className="w-4 h-4 text-cyan-400 fill-cyan-400" />
                <h3 className="text-white font-bold text-sm">Cosmic Optical Bench</h3>
            </div>
            <p className="text-gray-400 text-xs leading-relaxed font-mono">
                Visualizing the trajectory of photons from a background source (z=2.0), deflected by a massive cluster (z=0.5), arriving at Earth (z=0).
            </p>
        </div>
    </div>
  );
};

export default LensingProcess3D;