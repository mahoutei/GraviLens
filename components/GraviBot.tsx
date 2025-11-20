import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Loader2, Bot } from 'lucide-react';
import { Message } from '../types';
import { sendMessageToGraviBot } from '../services/geminiService';

const GraviBot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', content: 'Hello! I am GraviBot. Ask me anything about the GraviLens architecture, the SIE physics engine, or our dataset strategy.', timestamp: Date.now() }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) scrollToBottom();
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { role: 'user', content: input, timestamp: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    // Prepare history for API
    const history = messages.map(m => ({ role: m.role === 'model' ? 'model' : 'user', content: m.content }));
    
    const response = await sendMessageToGraviBot(history, userMsg.content);
    
    const botMsg: Message = { role: 'model', content: response, timestamp: Date.now() };
    setMessages(prev => [...prev, botMsg]);
    setIsLoading(false);
  };

  return (
    <>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={`fixed bottom-6 right-6 z-40 p-4 bg-cyan-600 hover:bg-cyan-500 text-white rounded-full shadow-lg shadow-cyan-900/50 transition-transform hover:scale-110 ${isOpen ? 'hidden' : 'flex'}`}
      >
        <Bot className="w-6 h-6" />
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-[350px] h-[500px] flex flex-col bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300">
          {/* Header */}
          <div className="bg-gray-800 p-4 flex justify-between items-center border-b border-gray-700">
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-cyan-400" />
              <h3 className="font-semibold text-white">GraviBot Assistant</h3>
            </div>
            <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-900/95">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3 rounded-lg text-sm ${
                  msg.role === 'user' 
                    ? 'bg-cyan-700 text-white rounded-br-none' 
                    : 'bg-gray-800 text-gray-200 rounded-bl-none border border-gray-700'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800 p-3 rounded-lg rounded-bl-none border border-gray-700">
                  <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-3 bg-gray-800 border-t border-gray-700 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask about the physics engine..."
              className="flex-1 bg-gray-950 text-white text-sm rounded-md px-3 py-2 border border-gray-700 focus:outline-none focus:border-cyan-500"
            />
            <button 
              onClick={handleSend}
              disabled={isLoading}
              className="bg-cyan-600 hover:bg-cyan-500 text-white p-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default GraviBot;