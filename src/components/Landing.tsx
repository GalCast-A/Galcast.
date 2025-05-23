import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import BackgroundChart from './BackgroundChart';
import TopNav from './TopNav';
import Footer from './Footer';

const Feature: React.FC<{ title: string; description: string; delay: number }> = ({ title, description, delay }) => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ duration: 0.8, delay }}
      className="dashboard-card backdrop-blur-sm bg-gradient-to-br from-card/80 to-card/40"
    >
      <h3 className="text-xl font-bold mb-3 text-accent">{title}</h3>
      <p className="text-gray-300">{description}</p>
    </motion.div>
  );
};

const Landing: React.FC = () => {
  const features = [
    {
      title: "Market Analysis",
      description: "Deep dives into market trends, technical analysis, and investment strategies from experienced traders and analysts."
    },
    {
      title: "Financial News Discussion",
      description: "Stay informed with the latest market news and join discussions about their impact on global markets."
    },
    {
      title: "Investment Strategies",
      description: "Share and learn about different investment approaches, from value investing to technical trading."
    },
    {
      title: "Community Insights",
      description: "Connect with fellow investors, share experiences, and discuss market opportunities and risks."
    }
  ];

  const [statsRef, statsInView] = useInView({
    triggerOnce: true,
    threshold: 0.1
  });

  return (
    <div className="min-h-screen bg-dark text-white overflow-hidden">
      <TopNav />
      
      <div className="relative z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-accent/5 to-transparent"></div>
        <BackgroundChart />
      </div>
      
      <div className="relative z-10">
        <div className="container mx-auto px-6 py-24">
          <div className="text-center max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              <div className="absolute inset-0 bg-dark/95 backdrop-blur-sm rounded-3xl"></div>
              <div className="relative p-8">
                <h1 className="text-5xl md:text-7xl font-bold mb-8 tracking-tight">
                  <span className="block text-white mb-4">Share Insights</span>
                  <div className="my-6">Understand Finance</div>
                  <span className="block text-white text-4xl md:text-6xl mt-6">
                    Navigate Markets
                  </span>
                </h1>

                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="text-xl text-gray-300 mb-12"
                >
                  Join our community of investors, traders, and analysts. Share insights, discuss market trends, 
                  and stay ahead of the financial markets.
                </motion.p>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                  className="flex flex-col md:flex-row items-center justify-center space-y-4 md:space-y-0 md:space-x-6"
                >
                  <Link to="/create" className="bg-gradient-to-r from-accent to-accent/80 text-black font-bold px-8 py-4 rounded-lg transform transition-all duration-300 hover:shadow-neon-gain glowing">
                    Share Your Analysis
                  </Link>
                  <Link to="/news" className="backdrop-blur-sm bg-white/5 border border-white/10 px-8 py-4 rounded-lg hover:bg-white/10 transition-all duration-300">
                    News & Sentiment
                  </Link>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>

        <div className="container mx-auto px-6 py-24">
          <motion.div
            ref={statsRef}
            initial={{ opacity: 0, y: 20 }}
            animate={statsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Join the Financial Discussion
            </h2>
            <p className="text-xl text-gray-300">
              Share knowledge, learn from others, and make informed investment decisions
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <Feature
                key={index}
                title={feature.title}
                description={feature.description}
                delay={0.2 * index}
              />
            ))}
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Landing;