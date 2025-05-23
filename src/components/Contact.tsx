import React from 'react';
import { motion } from 'framer-motion';
import TopNav from './TopNav';
import Footer from './Footer';

const Contact: React.FC = () => {
  return (
    <div className="min-h-screen bg-dark">
      <TopNav />
      
      <div className="container mx-auto px-6 pt-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto"
        >
          <div className="dashboard-card mb-8">
            <div className="flex flex-col md:flex-row gap-8 items-start">
              <div className="w-full md:w-1/3">
                <div className="aspect-square rounded-lg bg-gradient-to-br from-accent to-accent/40 flex items-center justify-center text-4xl font-bold text-dark">
                  AG
                </div>
              </div>
              <div className="w-full md:w-2/3">
                <h1 className="text-3xl font-bold text-white mb-4">
                  Hi, I'm Ander Galnares
                </h1>
                <p className="text-gray-300 mb-6">
                  Founder of Galcast.co, a free and interactive platform built to make portfolio analysis clear, accessible, and effective for everyday investors.
                </p>
                <div className="space-y-4 text-gray-300">
                  <p>
                    Currently studying at Boston College's Carroll School of Management, I'm passionate about making financial markets more accessible to everyone. My experience includes working with investment management teams and developing tools that help investors make data-driven decisions.
                  </p>
                  <p>
                    Through Galcast, I aim to bridge the gap between complex financial analysis and practical investment decisions. Whether you're a seasoned investor or just starting out, my goal is to provide you with clear, actionable insights backed by sophisticated analytics.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="dashboard-card"
          >
            <h2 className="text-2xl font-bold text-white mb-6">
              📫 Let's connect
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">Email</h3>
                  <a 
                    href="mailto:galnares@bc.edu"
                    className="text-accent hover:text-accent/80 transition-colors"
                  >
                    galnares@bc.edu
                  </a>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">LinkedIn</h3>
                  <a 
                    href="https://linkedin.com/in/ander-galnares"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:text-accent/80 transition-colors"
                  >
                    linkedin.com/in/ander-galnares
                  </a>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-1">Phone</h3>
                  <a 
                    href="tel:+528331880644"
                    className="text-accent hover:text-accent/80 transition-colors"
                  >
                    +52 833 188 0644
                  </a>
                </div>
              </div>
              <div className="bg-card/50 p-6 rounded-lg">
                <p className="text-gray-300 italic">
                  "I believe that everyone should have access to professional-grade financial analysis tools. Let's make smart investing accessible to all."
                </p>
                <p className="text-accent mt-4">- Ander</p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>

      <Footer />
    </div>
  );
};

export default Contact;