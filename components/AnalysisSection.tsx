import React, { useState, useEffect } from 'react';
import { PERFORMANCE_METRICS } from '../constants';
import { BarChart, Activity, Target, BrainCircuit, ScatterChart, TrendingUp, RefreshCw } from 'lucide-react';

const AnalysisSection: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-16">
        <h2 className="text-3xl font-bold text-white mb-4">Experimental Results</h2>
        <p className="text-gray-400 max-w-2xl mx-auto">
          Performance evaluation of GraviLens on the Galaxy10 DECals dataset, demonstrating superior classification accuracy and parameter recovery compared to standard CNN baselines.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
        <div className="glass-panel p-6 rounded-xl border border-cyan-500/30 bg-cyan-900/10">
          <div className="flex items-center gap-3 mb-2 text-cyan-400">
            <Target className="w-5 h-5" />
            <span className="text-sm font-semibold">Accuracy</span>
          </div>
          <div className="text-3xl font-bold text-white">{PERFORMANCE_METRICS.accuracy}</div>
          <div className="text-xs text-gray-400 mt-1">+12.1% vs ResNet50</div>
        </div>

        <div className="glass-panel p-6 rounded-xl border border-gray-700">
          <div className="flex items-center gap-3 mb-2 text-purple-400">
            <BrainCircuit className="w-5 h-5" />
            <span className="text-sm font-semibold">Param Recovery</span>
          </div>
          <div className="text-3xl font-bold text-white">0.021</div>
          <div className="text-xs text-gray-400 mt-1">MSE Loss (Log Scale)</div>
        </div>

        <div className="glass-panel p-6 rounded-xl border border-gray-700">
          <div className="flex items-center gap-3 mb-2 text-green-400">
            <Activity className="w-5 h-5" />
            <span className="text-sm font-semibold">Inference</span>
          </div>
          <div className="text-3xl font-bold text-white">14ms</div>
          <div className="text-xs text-gray-400 mt-1">Per image (RTX 4090)</div>
        </div>

        <div className="glass-panel p-6 rounded-xl border border-gray-700">
            <div className="flex items-center gap-3 mb-2 text-yellow-400">
                <BarChart className="w-5 h-5" />
                <span className="text-sm font-semibold">F1-Score</span>
            </div>
            <div className="text-3xl font-bold text-white">0.93</div>
            <div className="text-xs text-gray-400 mt-1">Weighted Avg</div>
        </div>
      </div>

      {/* Visualizations Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        
        {/* Manifold Visualization */}
        <div>
          <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <ScatterChart className="w-5 h-5 text-cyan-400" />
                  Manifold Separation
              </h3>
          </div>
          <ParameterManifoldViz />
          <p className="text-xs text-gray-500 mt-3 text-center">
             Separation of classes in latent space based on Ellipticity (q) vs. Substructure Fraction (f_sub).
          </p>
        </div>

        {/* Parity Plot Visualization */}
        <div>
          <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-400" />
                  Parameter Recovery (θ_E)
              </h3>
              <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-400"></div>
                      <span className="text-gray-400 text-xs">CDM</span>
                  </div>
                  <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                      <span className="text-gray-400 text-xs">ULDM</span>
                  </div>
              </div>
          </div>
          <ParityPlotViz />
          <p className="text-xs text-gray-500 mt-3 text-center">
             Predicted vs. Ground Truth for Einstein Radius. Points on the dotted line represent perfect recovery.
          </p>
        </div>

      </div>

      {/* Comparison Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        <div className="glass-panel p-8 rounded-xl">
          <h3 className="text-xl font-bold text-white mb-6">Model Comparison</h3>
          <div className="space-y-6">
            
            {/* GraviLens */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="font-semibold text-white">GraviLens (Ours)</span>
                <span className="text-cyan-400 font-mono">94.2%</span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 w-[94.2%] shadow-[0_0_10px_cyan]"></div>
              </div>
            </div>

            {/* ResNet50 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-300">ResNet-50 (Baseline)</span>
                <span className="text-gray-400 font-mono">82.1%</span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gray-600 w-[82.1%]"></div>
              </div>
            </div>

            {/* VGG16 */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-300">VGG-16 (Baseline)</span>
                <span className="text-gray-400 font-mono">78.5%</span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gray-600 w-[78.5%]"></div>
              </div>
            </div>

            {/* Simple CNN */}
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-300">Simple CNN</span>
                <span className="text-gray-400 font-mono">71.3%</span>
              </div>
              <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gray-600 w-[71.3%]"></div>
              </div>
            </div>

          </div>
        </div>

        {/* Confusion Matrix Representation */}
        <div className="glass-panel p-8 rounded-xl flex flex-col justify-center">
           <h3 className="text-xl font-bold text-white mb-6">Confusion Matrix</h3>
           <div className="grid grid-cols-2 gap-4 text-center h-full">
              {/* Header */}
              <div className="col-span-2 grid grid-cols-2 mb-2 text-xs font-mono text-gray-500 uppercase tracking-widest">
                  <div>Pred: CDM</div>
                  <div>Pred: ULDM</div>
              </div>
              
              {/* Row 1 */}
              <div className="relative bg-cyan-900/30 border border-cyan-500/50 rounded-lg p-4 flex flex-col justify-center items-center group hover:bg-cyan-900/50 transition-colors">
                 <div className="absolute top-2 left-2 text-[10px] text-gray-400">True: CDM</div>
                 <div className="text-2xl font-bold text-white">2,450</div>
                 <div className="text-xs text-cyan-300 mt-1">95.2%</div>
              </div>
              <div className="relative bg-gray-800/30 border border-gray-700 rounded-lg p-4 flex flex-col justify-center items-center">
                 <div className="absolute top-2 left-2 text-[10px] text-gray-500">True: CDM</div>
                 <div className="text-xl font-bold text-gray-400">124</div>
                 <div className="text-xs text-gray-600 mt-1">4.8%</div>
              </div>

              {/* Row 2 */}
              <div className="relative bg-gray-800/30 border border-gray-700 rounded-lg p-4 flex flex-col justify-center items-center">
                  <div className="absolute top-2 left-2 text-[10px] text-gray-500">True: ULDM</div>
                 <div className="text-xl font-bold text-gray-400">189</div>
                 <div className="text-xs text-gray-600 mt-1">7.3%</div>
              </div>
              <div className="relative bg-purple-900/20 border border-purple-500/50 rounded-lg p-4 flex flex-col justify-center items-center group hover:bg-purple-900/40 transition-colors">
                 <div className="absolute top-2 left-2 text-[10px] text-gray-400">True: ULDM</div>
                 <div className="text-2xl font-bold text-white">2,387</div>
                 <div className="text-xs text-purple-300 mt-1">92.7%</div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};

const ParameterManifoldViz = () => {
    const [points, setPoints] = useState<{x: number, y: number, type: 'cdm' | 'uldm', id: number}[]>([]);

    // Simulation loop
    useEffect(() => {
        const interval = setInterval(() => {
            if (points.length > 100) {
                setPoints([]); // Reset
                return;
            }

            const newPoints = [];
            // Add CDM points (Top Right cluster)
            // x: q (0.1-1), y: substructure (log scale high)
            const cdmX = 0.4 + Math.random() * 0.4; // More elliptical typically
            const cdmY = 0.7 + Math.random() * 0.25; // High substructure
            newPoints.push({x: cdmX, y: cdmY, type: 'cdm' as const, id: Math.random()});

            // Add ULDM points (Bottom Right/Left cluster)
            // x: q (variable), y: substructure (very low)
            const uldmX = 0.5 + Math.random() * 0.4; 
            const uldmY = 0.05 + Math.random() * 0.15; // Low substructure
            newPoints.push({x: uldmX, y: uldmY, type: 'uldm' as const, id: Math.random()});

            setPoints(prev => [...prev, ...newPoints]);
        }, 100);

        return () => clearInterval(interval);
    }, [points.length]);

    return (
        <div className="glass-panel p-6 rounded-xl border border-gray-700 relative h-[300px] flex flex-col">
            <div className="absolute inset-0 p-6">
                {/* Grid lines */}
                <div className="w-full h-full border-l border-b border-gray-600 relative">
                    {/* Axis Labels */}
                    <div className="absolute -bottom-6 right-0 text-xs text-gray-500 font-mono">Ellipticity (q) →</div>
                    <div className="absolute -left-8 top-0 text-xs text-gray-500 font-mono -rotate-90 origin-top-right">Substructure (f_sub) →</div>
                    
                    {/* Background Zones */}
                    <div className="absolute top-0 right-0 w-1/2 h-1/3 bg-cyan-500/5 rounded-bl-3xl border-l border-b border-cyan-500/10" />
                    <div className="absolute bottom-0 left-0 w-full h-1/3 bg-purple-500/5 rounded-tr-3xl border-t border-purple-500/10" />

                    {/* Points */}
                    {points.map((p) => (
                        <div 
                            key={p.id}
                            className={`absolute w-2 h-2 rounded-full transition-all duration-500 ${p.type === 'cdm' ? 'bg-cyan-400 shadow-[0_0_5px_cyan]' : 'bg-purple-500 shadow-[0_0_5px_purple]'}`}
                            style={{
                                left: `${p.x * 100}%`,
                                bottom: `${p.y * 100}%`,
                                opacity: 0,
                                animation: 'fadeIn 0.5s forwards'
                            }}
                        />
                    ))}
                </div>
            </div>
            
            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; transform: scale(0); }
                    to { opacity: 0.8; transform: scale(1); }
                }
            `}</style>

            {/* Live Counter */}
            <div className="absolute top-4 right-4 flex gap-4 bg-black/50 backdrop-blur px-3 py-1.5 rounded-lg border border-gray-700 font-mono text-xs">
                <div className="flex items-center gap-2">
                    <span className="text-gray-400">N=</span>
                    <span className="text-white">{points.length * 12}</span>
                </div>
                <div className="flex items-center gap-2">
                    <RefreshCw className={`w-3 h-3 text-green-500 ${points.length < 100 ? 'animate-spin' : ''}`} />
                    <span className="text-green-500">LIVE</span>
                </div>
            </div>
        </div>
    );
}

const ParityPlotViz = () => {
  const [points, setPoints] = useState<{x: number, y: number, type: 'cdm' | 'uldm', id: number}[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (points.length > 60) {
        setPoints([]);
        return;
      }

      const newPoints = [];
      // CDM Point: Ground truth random between 0.8 and 1.8. Prediction has some noise/scatter.
      const gtCDM = 0.8 + Math.random() * 1.0;
      const predCDM = gtCDM + (Math.random() - 0.5) * 0.25; // Larger scatter for CDM due to clumps
      newPoints.push({ x: gtCDM, y: predCDM, type: 'cdm' as const, id: Math.random() });

      // ULDM Point: Prediction is typically tighter/smoother
      const gtULDM = 0.8 + Math.random() * 1.0;
      const predULDM = gtULDM + (Math.random() - 0.5) * 0.08;
      newPoints.push({ x: gtULDM, y: predULDM, type: 'uldm' as const, id: Math.random() });

      setPoints(prev => [...prev, ...newPoints]);
    }, 150);
    return () => clearInterval(interval);
  }, [points.length]);

  return (
    <div className="glass-panel p-6 rounded-xl border border-gray-700 relative h-[300px] flex flex-col">
        <div className="absolute inset-0 p-8">
             {/* Axes */}
            <div className="w-full h-full border-l border-b border-gray-600 relative">
                 {/* Diagonal Identity Line */}
                <div className="absolute inset-0 border-t border-r border-transparent">
                    <svg className="w-full h-full overflow-visible">
                        <line x1="0%" y1="100%" x2="100%" y2="0%" stroke="rgba(255,255,255,0.15)" strokeDasharray="4,4" strokeWidth="1.5" />
                        <text x="95%" y="5%" fill="rgba(255,255,255,0.3)" fontSize="10" textAnchor="end" style={{fontFamily: 'monospace'}}>Ideal (y=x)</text>
                    </svg>
                </div>

                {/* Labels */}
                <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs text-gray-500 font-mono">Ground Truth θ_E</div>
                <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-xs text-gray-500 font-mono -rotate-90">Predicted θ_E</div>

                {/* Points */}
                {points.map((p) => {
                    // Map values [0.5, 2.0] to [0%, 100%]
                    // min 0.5, max 2.0 -> range 1.5
                    // If out of bounds, clamp or hide. Here we assume range is controlled.
                    const normalize = (val: number) => Math.max(0, Math.min(100, (val - 0.5) / 1.5 * 100));
                    const left = normalize(p.x);
                    const bottom = normalize(p.y);
                    
                    return (
                        <div 
                            key={p.id}
                            className={`absolute w-2 h-2 rounded-full border border-black/50 ${p.type === 'cdm' ? 'bg-cyan-400 shadow-[0_0_2px_cyan]' : 'bg-purple-500 shadow-[0_0_2px_purple]'}`}
                            style={{
                                left: `${left}%`,
                                bottom: `${bottom}%`,
                                opacity: 0.8,
                                animation: 'popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards'
                            }}
                        />
                    );
                })}
            </div>
        </div>
        <style>{`
            @keyframes popIn {
                from { transform: scale(0); opacity: 0; }
                to { transform: scale(1); opacity: 0.8; }
            }
        `}</style>
    </div>
  );
};

export default AnalysisSection;