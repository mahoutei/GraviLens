import React from 'react';
import { TIMELINE_DATA } from '../constants';
import { CheckCircle, Clock, Circle } from 'lucide-react';

const Timeline: React.FC = () => {
  return (
    <div className="relative border-l border-gray-700 ml-3 md:ml-6 space-y-12">
      {TIMELINE_DATA.map((phase, index) => (
        <div key={index} className="mb-10 ml-6">
          <span className={`absolute flex items-center justify-center w-8 h-8 rounded-full -left-4 ring-4 ring-gray-900 ${
            phase.status === 'complete' ? 'bg-green-500' : 
            phase.status === 'in-progress' ? 'bg-cyan-500' : 'bg-gray-700'
          }`}>
            {phase.status === 'complete' ? <CheckCircle className="w-5 h-5 text-white" /> :
             phase.status === 'in-progress' ? <ActivityIcon className="w-5 h-5 text-white animate-spin-slow" /> :
             <Circle className="w-5 h-5 text-gray-400" />}
          </span>
          
          <div className="glass-panel p-6 rounded-lg border border-gray-700 hover:border-cyan-500/30 transition-colors">
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-4">
              <h3 className="flex items-center text-lg font-semibold text-white">
                {phase.phase}: {phase.title}
              </h3>
              <span className="text-sm font-mono text-cyan-400 bg-cyan-950/30 px-2 py-1 rounded mt-2 md:mt-0 inline-block w-fit">
                {phase.date}
              </span>
            </div>
            
            <ul className="space-y-3">
              {phase.items.map((item, idx) => {
                const isDone = item.includes("[COMPLETE]");
                const isProgress = item.includes("[IN PROGRESS]");
                return (
                  <li key={idx} className="flex items-start text-gray-300 text-sm">
                    <span className={`mr-2 mt-1.5 w-1.5 h-1.5 rounded-full ${
                      isDone ? 'bg-green-500' : isProgress ? 'bg-cyan-400 animate-pulse' : 'bg-gray-600'
                    }`} />
                    {item.replace(/\[.*?\]/, '')}
                    {isDone && <span className="ml-2 text-xs text-green-500 font-mono border border-green-900 bg-green-900/20 px-1 rounded">DONE</span>}
                    {isProgress && <span className="ml-2 text-xs text-cyan-400 font-mono border border-cyan-900 bg-cyan-900/20 px-1 rounded">ACTIVE</span>}
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
};

const ActivityIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

export default Timeline;