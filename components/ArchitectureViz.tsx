import React, { useState, useEffect } from 'react';
import { Box, Cpu, Activity, Eye, Binary, Zap, Database, ScanLine } from 'lucide-react';

const ArchitectureViz: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  const steps = [
    {
      id: 'input',
      title: "Input Image",
      subtitle: "Lensed Galaxy (64x64)",
      icon: <Box className="w-6 h-6" />,
      description: "The pipeline ingests a 64x64 pixel single-channel tensor representing a gravitationally lensed galaxy. The image contains distorted arcs caused by a foreground Dark Matter halo.",
      color: "text-purple-400",
      borderColor: "border-purple-500",
      bgGradient: "from-purple-500/20 to-transparent"
    },
    {
      id: 'backbone',
      title: "Transformer Backbone",
      subtitle: "Hierarchical Features",
      icon: <Cpu className="w-6 h-6" />,
      description: "A Vision Transformer backbone extracts features using hierarchical attention. This mechanism allows the network to attend to local substructures (clumps) and global lensing arcs simultaneously.",
      color: "text-blue-400",
      borderColor: "border-blue-500",
      bgGradient: "from-blue-500/20 to-transparent"
    },
    {
      id: 'params',
      title: "Parameter Head",
      subtitle: "Physics Prediction",
      icon: <Activity className="w-6 h-6" />,
      description: "The network predicts the macroscopic parameters of the lens: Einstein Radius (k), Ellipticity (q), and Orientation (φ). These define the gravitational potential profile.",
      color: "text-cyan-400",
      borderColor: "border-cyan-500",
      bgGradient: "from-cyan-500/20 to-transparent"
    },
    {
      id: 'physics',
      title: "Physics Engine",
      subtitle: "Relativistic Encoder",
      icon: <Zap className="w-6 h-6" />,
      description: "StableSIEDeflection: A differentiable physics layer calculates the exact deflection field α(x,y) based on General Relativity, simulating how light bends around the predicted mass.",
      color: "text-green-400",
      borderColor: "border-green-500",
      bgGradient: "from-green-500/20 to-transparent"
    },
    {
      id: 'recon',
      title: "Reconstruction",
      subtitle: "Inverse Ray-Trace",
      icon: <Eye className="w-6 h-6" />,
      description: "Using the calculated deflection field, the model 'un-lenses' the input image via differentiable sampling. If the physics parameters are correct, the original source galaxy is revealed.",
      color: "text-yellow-400",
      borderColor: "border-yellow-500",
      bgGradient: "from-yellow-500/20 to-transparent"
    },
    {
      id: 'class',
      title: "Classification",
      subtitle: "CDM vs ULDM",
      icon: <Binary className="w-6 h-6" />,
      description: "The final classification head analyzes both the reconstructed source features and the physics parameter manifold to distinguish between Cold Dark Matter (particle) and Ultra-Light Dark Matter (wave).",
      color: "text-pink-400",
      borderColor: "border-pink-500",
      bgGradient: "from-pink-500/20 to-transparent"
    }
  ];

  useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 4000); // Slower interval to enjoy animations
    return () => clearInterval(interval);
  }, [isAutoPlaying, steps.length]);

  const handleStepClick = (index: number) => {
    setActiveStep(index);
    setIsAutoPlaying(false);
    setTimeout(() => setIsAutoPlaying(true), 15000);
  };

  return (
    <div className="w-full flex flex-col gap-8">
      <style>{`
        @keyframes scan {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.8); opacity: 0.5; }
          100% { transform: scale(1.2); opacity: 0; }
        }
        @keyframes dash {
          to { stroke-dashoffset: 0; }
        }
        .animate-scan { animation: scan 2s linear infinite; }
        .animate-pulse-ring { animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .animate-dash { stroke-dasharray: 10; stroke-dashoffset: 100; animation: dash 1.5s linear infinite; }
      `}</style>

      {/* Pipeline Progress Bar */}
      <div className="relative px-4">
        <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-gray-800 -translate-y-1/2 rounded-full" />
        <div 
          className="hidden md:block absolute top-1/2 left-0 h-1 bg-gradient-to-r from-cyan-500 to-pink-500 -translate-y-1/2 rounded-full transition-all duration-500 ease-in-out"
          style={{ width: `${(activeStep / (steps.length - 1)) * 100}%` }}
        />
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 relative z-10">
          {steps.map((step, index) => {
            const isActive = activeStep === index;
            const isCompleted = activeStep > index;
            return (
              <div 
                key={step.id}
                onClick={() => handleStepClick(index)}
                className={`
                  group cursor-pointer flex flex-col items-center text-center p-2 rounded-xl border transition-all duration-300
                  ${isActive ? `bg-gray-900 ${step.borderColor} shadow-[0_0_15px_rgba(0,0,0,0.5)] scale-105 border-b-4` : 
                    isCompleted ? 'bg-gray-900/50 border-gray-700 opacity-70' : 'bg-gray-900/20 border-transparent opacity-40 hover:opacity-80'}
                `}
              >
                <div className={`w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center mb-2 transition-colors ${isActive ? step.color : 'text-gray-500'}`}>
                  {step.icon}
                </div>
                <h4 className={`text-xs font-bold ${isActive ? 'text-white' : 'text-gray-400'}`}>{step.title}</h4>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Step Detailed View */}
      <div className="w-full bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-2xl overflow-hidden shadow-2xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 min-h-[350px]">
          
          {/* Left: Visual Animation */}
          <div className="bg-black/40 flex items-center justify-center p-8 border-b lg:border-b-0 lg:border-r border-gray-800 relative overflow-hidden">
            <div className={`absolute inset-0 bg-gradient-to-br ${steps[activeStep].bgGradient} opacity-10`} />
            
            {/* Render Specific Visualization */}
            <div className="relative z-10 w-full max-w-[300px] aspect-square flex items-center justify-center">
               {renderVisual(steps[activeStep].id)}
            </div>
          </div>

          {/* Right: Description */}
          <div className="p-8 flex flex-col justify-center relative">
            <div className={`absolute top-0 right-0 p-6 opacity-10 ${steps[activeStep].color}`}>
              {React.cloneElement(steps[activeStep].icon as React.ReactElement<{ className?: string }>, { className: "w-32 h-32" })}
            </div>
            
            <div className="relative z-10 animate-in slide-in-from-right-4 duration-500" key={activeStep}>
              <span className={`inline-block px-3 py-1 rounded-full text-xs font-mono mb-4 border ${steps[activeStep].borderColor} ${steps[activeStep].color} bg-opacity-10 bg-white`}>
                Step 0{activeStep + 1}
              </span>
              <h3 className="text-3xl font-bold text-white mb-2">{steps[activeStep].title}</h3>
              <h4 className={`text-xl ${steps[activeStep].color} mb-6 font-mono`}>{steps[activeStep].subtitle}</h4>
              <p className="text-gray-300 leading-relaxed text-lg">{steps[activeStep].description}</p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

// --- Visualization Sub-Components ---

const InputVisual = () => (
  <div className="relative w-48 h-48 group">
    <div className="absolute inset-0 bg-gray-950 border border-gray-700 rounded-lg overflow-hidden">
      <img 
        src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/A_Horseshoe_Einstein_Ring_from_Hubble.JPG/600px-A_Horseshoe_Einstein_Ring_from_Hubble.JPG" 
        alt="Gravitational Lens" 
        className="w-full h-full object-cover opacity-80 group-hover:scale-110 transition-transform duration-700"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent to-transparent opacity-60"></div>
    </div>
    <div className="absolute -bottom-6 left-0 right-0 text-center text-xs font-mono text-purple-400">Tensor: [1, 64, 64]</div>
  </div>
);

const BackboneVisual = () => (
  <div className="relative w-48 h-48 flex items-center justify-center">
    <div className="grid grid-cols-4 gap-1 w-full h-full p-2 bg-gray-900 border border-gray-700 rounded-lg">
      {[...Array(16)].map((_, i) => (
        <div key={i} className="bg-blue-900/20 rounded-sm border border-blue-500/10"></div>
      ))}
    </div>
    {/* Moving Attention Window */}
    <div className="absolute w-24 h-24 border-2 border-blue-400 bg-blue-400/10 shadow-[0_0_15px_rgba(59,130,246,0.5)] top-2 left-2 animate-[scan_4s_steps(2)_infinite] rounded-md backdrop-blur-sm flex items-center justify-center">
        <Cpu className="w-8 h-8 text-blue-400 animate-spin-slow opacity-50" />
    </div>
    <div className="absolute -bottom-6 left-0 right-0 text-center text-xs font-mono text-blue-400">Hierarchical Attention</div>
  </div>
);

const ParamsVisual = () => (
  <div className="w-56 bg-gray-900 border border-cyan-900/50 rounded-xl p-4 flex flex-col gap-3 shadow-[0_0_20px_rgba(8,145,178,0.2)]">
    <div className="flex justify-between items-center border-b border-gray-800 pb-2">
      <span className="text-gray-400 text-sm font-mono">θ_E (k)</span>
      <span className="text-cyan-400 font-mono font-bold animate-pulse">1.42</span>
    </div>
    <div className="flex justify-between items-center border-b border-gray-800 pb-2">
      <span className="text-gray-400 text-sm font-mono">q (Shape)</span>
      <span className="text-cyan-400 font-mono font-bold animate-pulse" style={{animationDelay: '0.3s'}}>0.65</span>
    </div>
    <div className="flex justify-between items-center">
      <span className="text-gray-400 text-sm font-mono">φ (Angle)</span>
      <span className="text-cyan-400 font-mono font-bold animate-pulse" style={{animationDelay: '0.6s'}}>45.0°</span>
    </div>
  </div>
);

const PhysicsVisual = () => (
  <div className="relative w-full h-full flex items-center justify-center">
    <svg viewBox="0 0 200 200" className="w-48 h-48 overflow-visible">
      {/* Grid Warp */}
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1"/>
        </pattern>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      
      <rect width="200" height="200" fill="url(#grid)" />
      
      {/* Lens Mass */}
      <circle cx="100" cy="100" r="15" fill="#22c55e" filter="url(#glow)" className="animate-pulse">
        <animate attributeName="r" values="15;18;15" dur="3s" repeatCount="indefinite" />
      </circle>

      {/* Light Rays */}
      <path d="M 0 50 Q 100 80 200 50" stroke="#4ade80" strokeWidth="2" fill="none" className="animate-dash" opacity="0.7" />
      <path d="M 0 150 Q 100 120 200 150" stroke="#4ade80" strokeWidth="2" fill="none" className="animate-dash" opacity="0.7" style={{animationDelay: '0.5s'}} />
    </svg>
    <div className="absolute -bottom-6 w-full text-center text-xs font-mono text-green-400">α(x,y) Calculation</div>
  </div>
);

const ReconVisual = () => (
  <div className="relative w-48 h-48 flex items-center justify-center">
    <div className="absolute inset-0 border border-dashed border-yellow-600/50 rounded-full animate-spin-slow duration-[10s]"></div>
    
    {/* Source Galaxy (Reconstructed) */}
    <div className="relative w-24 h-24">
       {/* Core */}
       <div className="absolute inset-0 bg-yellow-500/20 blur-xl rounded-full animate-pulse"></div>
       <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-[0_0_10px_white]"></div>
       
       {/* Spiral Arms (CSS) */}
       <div className="absolute inset-0 border-t-4 border-r-4 border-yellow-400 rounded-full opacity-80" style={{transform: 'rotate(45deg)'}}></div>
       <div className="absolute inset-0 border-b-4 border-l-4 border-yellow-400 rounded-full opacity-80" style={{transform: 'rotate(225deg) scale(0.7)'}}></div>
    </div>
    
    {/* Scan line effect */}
    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-yellow-400/10 to-transparent h-full w-full animate-scan"></div>
    <div className="absolute -bottom-6 w-full text-center text-xs font-mono text-yellow-400">Source Plane β</div>
  </div>
);

const ClassVisual = () => (
  <div className="w-56 flex flex-col gap-4">
    <div className="space-y-2">
      <div className="flex justify-between text-xs font-mono text-gray-400">
        <span>CDM (Particle)</span>
        <span>92.4%</span>
      </div>
      <div className="h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
        <div className="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 w-[92.4%] shadow-[0_0_10px_cyan] animate-[width_1s_ease-out]"></div>
      </div>
    </div>

    <div className="space-y-2">
      <div className="flex justify-between text-xs font-mono text-gray-400">
        <span>ULDM (Wave)</span>
        <span>7.6%</span>
      </div>
      <div className="h-3 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
        <div className="h-full bg-purple-600 w-[7.6%]"></div>
      </div>
    </div>
    
    <div className="mt-2 p-2 bg-cyan-900/20 border border-cyan-500/30 rounded text-center">
      <span className="text-cyan-400 font-bold text-sm tracking-wider animate-pulse">CDM DETECTED</span>
    </div>
  </div>
);

const renderVisual = (stepId: string) => {
  switch(stepId) {
    case 'input': return <InputVisual />;
    case 'backbone': return <BackboneVisual />;
    case 'params': return <ParamsVisual />;
    case 'physics': return <PhysicsVisual />;
    case 'recon': return <ReconVisual />;
    case 'class': return <ClassVisual />;
    default: return null;
  }
};

export default ArchitectureViz;