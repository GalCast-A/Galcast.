import React from 'react';
import { motion } from 'framer-motion';
import { Line, Pie } from 'react-chartjs-2';
import TopNav from './TopNav';
import Footer from './Footer';

interface ResultsProps {
  onBack: () => void;
  onNavigate: (view: string) => void;
  result: any;
}

const Results: React.FC<ResultsProps> = ({ onBack, onNavigate, result }) => {
  return (
    <div className="min-h-screen bg-dark">
      <TopNav onNavigate={onNavigate} />
      
      <div className="container mx-auto px-6 pt-20 pb-16">
        <motion.button
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={onBack}
          className="mb-8 px-4 py-2 text-white hover:bg-white/10 rounded-lg transition-colors flex items-center space-x-2"
        >
          <span>←</span>
          <span>Back to Portfolio Input</span>
        </motion.button>

        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold text-white mb-8"
        >
          Analysis Results
        </motion.h1>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-8"
        >
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <motion.div 
              whileHover={{ scale: 1.02 }}
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10 hover:border-accent/30 transition-all"
            >
              <h3 className="text-sm font-medium text-gray-400">Annual Return</h3>
              <p className="text-2xl font-bold text-accent mt-2">
                {(result.optimized_metrics?.annual_return * 100).toFixed(2)}%
              </p>
            </motion.div>
            <motion.div 
              whileHover={{ scale: 1.02 }}
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10 hover:border-accent/30 transition-all"
            >
              <h3 className="text-sm font-medium text-gray-400">Sharpe Ratio</h3>
              <p className="text-2xl font-bold text-accent mt-2">
                {result.optimized_metrics?.sharpe_ratio.toFixed(2)}
              </p>
            </motion.div>
            <motion.div 
              whileHover={{ scale: 1.02 }}
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10 hover:border-accent/30 transition-all"
            >
              <h3 className="text-sm font-medium text-gray-400">Max Drawdown</h3>
              <p className="text-2xl font-bold text-accent mt-2">
                {(result.optimized_metrics?.maximum_drawdown * 100).toFixed(2)}%
              </p>
            </motion.div>
            <motion.div 
              whileHover={{ scale: 1.02 }}
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10 hover:border-accent/30 transition-all"
            >
              <h3 className="text-sm font-medium text-gray-400">Value at Risk</h3>
              <p className="text-2xl font-bold text-accent mt-2">
                {(result.optimized_metrics?.value_at_risk * 100).toFixed(2)}%
              </p>
            </motion.div>
          </div>

          {/* Performance Chart */}
          <motion.div 
            className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-xl font-semibold mb-4 text-white">Performance vs Benchmark</h2>
            {result.cumulative_returns && (
              <div className="h-[400px]">
                <Line
                  data={{
                    labels: Object.values(result.cumulative_returns)[0]?.dates || [],
                    datasets: Object.entries(result.cumulative_returns || {}).map(([label, series]: [string, any], index) => {
                      const values = Array.isArray(series?.values) ? series.values : [];
                      return {
                        label,
                        data: values.map((v: number) => (typeof v === 'number' ? v * 100 : 0)),
                        borderColor: index === 0 ? '#00B050' : '#64748b',
                        backgroundColor: index === 0 ? 'rgba(0, 176, 80, 0.1)' : 'rgba(100, 116, 139, 0.1)',
                        tension: 0.4,
                        fill: true
                      };
                    })
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top',
                        labels: { color: 'white' }
                      },
                      tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(0, 176, 80, 0.2)',
                        borderWidth: 1
                      }
                    },
                    scales: {
                      x: {
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: { color: 'white' }
                      },
                      y: {
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: { 
                          color: 'white',
                          callback: (value) => `${value}%`
                        }
                      }
                    }
                  }}
                />
              </div>
            )}
          </motion.div>

          {/* Portfolio Composition */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div 
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-xl font-semibold mb-4 text-white">Current Allocation</h2>
              <Pie
                data={{
                  labels: result.portfolio_exposures?.original?.labels || [],
                  datasets: [{
                    data: result.portfolio_exposures?.original?.exposures?.map((v: number) => v * 100) || [],
                    backgroundColor: [
                      '#00B050',
                      '#008F40',
                      '#006E30',
                      '#004D20'
                    ]
                  }]
                }}
                options={{
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: { color: 'white' }
                    }
                  }
                }}
              />
            </motion.div>
            <motion.div 
              className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-xl font-semibold mb-4 text-white">Optimized Allocation</h2>
              <Pie
                data={{
                  labels: result.portfolio_exposures?.optimized?.labels || [],
                  datasets: [{
                    data: result.portfolio_exposures?.optimized?.exposures?.map((v: number) => v * 100) || [],
                    backgroundColor: [
                      '#00B050',
                      '#008F40',
                      '#006E30',
                      '#004D20'
                    ]
                  }]
                }}
                options={{
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: { color: 'white' }
                    }
                  }
                }}
              />
            </motion.div>
          </div>

          {/* Factor Analysis */}
          <motion.div 
            className="dashboard-card bg-card/90 backdrop-blur-sm border border-accent/10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h2 className="text-xl font-semibold mb-4 text-white">Factor Analysis</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-white">
                <thead>
                  <tr className="border-b border-accent/20">
                    <th className="text-left py-3 px-4">Factor</th>
                    <th className="text-left py-3 px-4">Exposure</th>
                  </tr>
                </thead>
                <tbody>
                  {result.fama_french_exposures && Object.entries(result.fama_french_exposures).map(([factor, exposure]: [string, any]) => (
                    <tr key={factor} className="border-b border-accent/10 hover:bg-accent/5">
                      <td className="py-3 px-4">{factor}</td>
                      <td className="py-3 px-4">{exposure.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </motion.div>
      </div>

      <Footer onNavigate={onNavigate} />
    </div>
  );
};

export default Results;