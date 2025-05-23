import React from 'react';
import { Link } from 'react-router-dom';

const Hero: React.FC = () => {
  return (
    <div className="relative min-h-screen flex items-center justify-center bg-darker overflow-hidden pt-16">
      <div className="absolute inset-0 z-0">
        <div className="h-full w-full bg-grid">
          <div className="absolute inset-0 bg-gradient-radial from-darker to-transparent opacity-90"></div>
        </div>
      </div>

      <div className="container mx-auto px-4 relative z-10 text-center">
        <div className="floating">
          <h1 className="text-4xl md:text-7xl font-bold mb-8 text-white leading-tight tracking-tight">
            Share Insights<br />
            <span className="typing-text relative inline-block my-4">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent via-accent/80 to-accent">
                Understand Finance
              </span>
            </span><br />
            Navigate Markets
          </h1>
        </div>
        
        <p className="text-gray-400 text-lg md:text-xl max-w-2xl mx-auto mb-12 leading-relaxed">
          Join our community of investors, traders, and analysts. Share insights, discuss market trends, 
          and stay ahead of the financial markets.
        </p>
        
        <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-6 relative z-20">
          <Link 
            to="/posts" 
            className="bg-gradient-to-r from-accent to-accent/80 text-black font-bold px-8 py-4 rounded-lg transform transition-all duration-300 hover:shadow-neon-gain glowing"
          >
            Blogs & Posts
          </Link>
          <Link 
            to="/create" 
            className="backdrop-blur-sm bg-white/5 border border-white/10 px-8 py-4 rounded-lg hover:bg-white/10 transition-all duration-300"
          >
            Share Your Analysis
          </Link>
        </div>
        
        <div className="mt-16">
          <span className="glass-effect text-gain px-8 py-4 rounded-full text-sm font-medium inline-flex items-center space-x-2">
            <span className="w-2 h-2 bg-gain rounded-full animate-pulse"></span>
            <span>Corporate Analytics Solutions • Accessible to All</span>
          </span>
        </div>
      </div>
    </div>
  );
};

export default Hero;