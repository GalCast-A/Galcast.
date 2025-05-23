import React from 'react';
import { Link } from 'react-router-dom';

const Footer: React.FC = () => {
  return (
    <footer className="bg-card/80 backdrop-blur-md border-t border-border/20 py-12">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          <div>
            <Link to="/" className="text-xl font-bold text-accent mb-4 block">
              Galcast.co
            </Link>
            <p className="text-gray-400">
              Advanced portfolio analytics and optimization tools for modern investors.
            </p>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-white mb-4">Navigation</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <Link to="/posts" className="hover:text-accent transition-colors">
                  Blog Posts
                </Link>
              </li>
              <li>
                <Link to="/news" className="hover:text-accent transition-colors">
                  News & Sentiment
                </Link>
              </li>
              <li>
                <Link to="/create" className="hover:text-accent transition-colors">
                  Share Analysis
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-white mb-4">Resources</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <Link to="/disclaimer" className="hover:text-accent transition-colors">
                  Disclaimer
                </Link>
              </li>
              <li>
                <Link to="/privacy" className="hover:text-accent transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/terms" className="hover:text-accent transition-colors">
                  Terms of Service
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-white mb-4">Connect</h4>
            <ul className="space-y-2 text-gray-400">
              <li>
                <a href="mailto:galnares@bc.edu" className="hover:text-accent transition-colors">
                  Email
                </a>
              </li>
              <li>
                <a href="https://linkedin.com/in/ander-galnares" target="_blank" rel="noopener noreferrer" className="hover:text-accent transition-colors">
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-border/20 pt-8">
          <div className="text-sm text-gray-400">
            <p>
              The content and tools on this site are for informational purposes only and do not constitute financial, investment, or legal advice. 
              Nothing here should be considered a recommendation. Use at your own risk. Always consult a licensed financial advisor.
            </p>
            <div className="mt-6">
              © {new Date().getFullYear()} Galcast Analytics. All rights reserved.
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;