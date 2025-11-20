import React from 'react';
import Navbar from './components/Navbar';
import LensingBackground from './components/LensingBackground';
import GraviBot from './components/GraviBot';
import ArchitectureViz from './components/ArchitectureViz';
import PhysicsEngineDemo from './components/PhysicsEngineDemo';
import AnalysisSection from './components/AnalysisSection';
import LensingProcess3D from './components/LensingProcess3D';
import { SectionId } from './types';
import { PROJECT_INFO } from './constants';
import { Download, Github, Database } from 'lucide-react';

const App: React.FC = () => {
  return (
    <div className="relative min-h-screen">
      <LensingBackground />
      <Navbar />
      
      <main className="relative z-10 pt-16">
        {/* Hero Section */}
        <section id={SectionId.HOME} className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 relative">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-block mb-4 px-3 py-1 rounded-full bg-cyan-900/30 border border-cyan-500/30 text-cyan-400 text-sm font-mono">
              v{PROJECT_INFO.version} | {PROJECT_INFO.date}
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold text-white tracking-tight mb-6 drop-shadow-2xl">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-white to-cyan-400">
                GraviLens
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 font-light leading-relaxed">
              Unveiling the nature of Dark Matter through <span className="text-cyan-300 font-medium">Physics-Informed Neural Networks</span> and Strong Gravitational Lensing.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-3 bg-white text-gray-900 rounded-full font-semibold hover:bg-gray-200 transition-all flex items-center justify-center gap-2">
                <Download className="w-5 h-5" /> Read Documentation
              </button>
              <button className="px-8 py-3 bg-gray-800/50 backdrop-blur-sm border border-gray-700 text-white rounded-full font-semibold hover:bg-gray-800 transition-all flex items-center justify-center gap-2">
                <Github className="w-5 h-5" /> View Source
              </button>
            </div>
          </div>
          
          <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 animate-bounce text-gray-500">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </section>

        {/* Summary Section */}
        <section id={SectionId.SUMMARY} className="py-24 px-4 sm:px-6 lg:px-8 bg-black/30">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
              <div>
                <h2 className="text-3xl font-bold text-white mb-6 border-l-4 border-cyan-500 pl-4">Executive Summary</h2>
                <p className="text-gray-300 text-lg leading-relaxed mb-6">
                  <strong className="text-white">GraviLens</strong> is a novel deep learning architecture designed to classify Dark Matter morphology—specifically distinguishing between <span className="text-cyan-300">Cold Dark Matter (CDM)</span> and <span className="text-purple-300">Ultra-Light Dark Matter (ULDM)</span>.
                </p>
                <p className="text-gray-400 leading-relaxed">
                  Unlike traditional CNNs, this project utilizes a <strong>Physics-Informed Neural Network (PINN)</strong> approach. It embeds a differentiable, analytical physics engine based on General Relativity directly into the model’s computation graph, allowing for "inverse ray-tracing" of lensed images.
                </p>
              </div>
              <div className="glass-panel p-8 rounded-2xl">
                <h3 className="text-xl font-bold text-white mb-4">Core Innovation</h3>
                <ul className="space-y-4">
                  <li className="flex items-start">
                    <div className="flex-shrink-0 h-6 w-6 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 mt-1">1</div>
                    <div className="ml-4">
                      <p className="text-white font-medium">Relativistic Encoder</p>
                      <p className="text-sm text-gray-400">Functions as a differentiable simulator within the network.</p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <div className="flex-shrink-0 h-6 w-6 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 mt-1">2</div>
                    <div className="ml-4">
                      <p className="text-white font-medium">Transformer Backbone</p>
                      <p className="text-sm text-gray-400">Captures both local sub-halos and global gravitational arcs.</p>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <div className="flex-shrink-0 h-6 w-6 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 mt-1">3</div>
                    <div className="ml-4">
                      <p className="text-white font-medium">Physics-Informed Loss</p>
                      <p className="text-sm text-gray-400">Recovers physical parameters (κ, q, φ) alongside classification.</p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Architecture Section */}
        <section id={SectionId.ARCHITECTURE} className="py-24 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl font-bold text-white mb-4">Deep Learning Architecture</h2>
              <p className="text-gray-400 max-w-2xl mx-auto">
                An end-to-end physics-informed pipeline that fuses Vision Transformer features with a relativistic lensing engine.
              </p>
            </div>
            
            <div className="mb-8">
               <ArchitectureViz />
            </div>
          </div>
        </section>

        {/* Physics Section */}
        <section id={SectionId.PHYSICS} className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-transparent to-gray-900/80">
          <div className="max-w-7xl mx-auto space-y-16">
             <div>
                 <h2 className="text-3xl font-bold text-white mb-8 text-center">Mathematical Framework</h2>
                 <p className="text-gray-400 text-center max-w-2xl mx-auto mb-8">
                   Gravitational lensing is governed by General Relativity. Light rays from a source at position β are deflected by an angle α to appear at position θ.
                 </p>
                 <LensingProcess3D />
             </div>
             
             <div className="pt-8 border-t border-gray-800">
                <h3 className="text-2xl font-bold text-white mb-6">Interactive Simulator</h3>
                <PhysicsEngineDemo />
             </div>
          </div>
        </section>

        {/* Analysis & Results Section (NEW) */}
        <section id={SectionId.ANALYSIS} className="py-24 px-4 sm:px-6 lg:px-8 bg-black/40">
          <AnalysisSection />
        </section>

        {/* Dataset Section */}
        <section id={SectionId.DATASET} className="py-24 px-4 sm:px-6 lg:px-8">
          <div className="max-w-7xl mx-auto">
            <h2 className="text-3xl font-bold text-white mb-12 text-center">Dataset Generation Strategy</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="glass-panel p-8 rounded-xl border-t-4 border-purple-500">
                <h3 className="text-2xl font-bold text-white mb-4">ULDM Generation</h3>
                <div className="space-y-4">
                  <div className="flex justify-between border-b border-gray-700 pb-2">
                    <span className="text-gray-400">Mass (k)</span>
                    <span className="font-mono text-purple-300">1.0 ± 0.1 (Moderate)</span>
                  </div>
                  <div className="flex justify-between border-b border-gray-700 pb-2">
                    <span className="text-gray-400">Shape (q)</span>
                    <span className="font-mono text-purple-300">0.85 ± 0.1 (Smooth)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Substructure</span>
                    <span className="font-mono text-purple-300">None (Wave-like)</span>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-8 rounded-xl border-t-4 border-cyan-500">
                <h3 className="text-2xl font-bold text-white mb-4">CDM Generation</h3>
                <div className="space-y-4">
                  <div className="flex justify-between border-b border-gray-700 pb-2">
                    <span className="text-gray-400">Mass (k)</span>
                    <span className="font-mono text-cyan-300">1.4 ± 0.2 (Stronger)</span>
                  </div>
                  <div className="flex justify-between border-b border-gray-700 pb-2">
                    <span className="text-gray-400">Shape (q)</span>
                    <span className="font-mono text-cyan-300">0.5 ± 0.2 (Triaxial)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Substructure</span>
                    <span className="font-mono text-cyan-300">Clumps (Particle-like)</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-8 text-center">
              <p className="text-gray-500 text-sm flex items-center justify-center gap-2">
                <Database className="w-4 h-4" /> Source: Galaxy10 DECals (Real galaxy images) simulated via lenstronomy + pyHalo.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-gray-950 border-t border-gray-800 py-12 text-center">
          <p className="text-gray-500 text-sm">© 2025 GraviLens Project. All Rights Reserved.</p>
          <p className="text-gray-600 text-xs mt-2">Documentation Version 2.01-STABLE</p>
        </footer>
      </main>
      
      <GraviBot />
    </div>
  );
};

export default App;
