import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ChartBarSquareIcon } from '@heroicons/react/24/solid';
import { useUser, useSupabaseClient } from '@supabase/auth-helpers-react';
import toast from 'react-hot-toast';

const TopNav: React.FC = () => {
  const user = useUser();
  const supabase = useSupabaseClient();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      toast.error('Error signing out');
    } else {
      toast.success('Signed out successfully');
      navigate('/');
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-dark/80 backdrop-blur-md border-b border-border/20">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3 cursor-pointer"
          >
            <Link to="/" className="flex items-center space-x-3">
              <div className="relative w-10 h-10 bg-gradient-to-br from-accent to-accent/80 rounded shadow-neon-gain overflow-hidden transform hover:scale-105 transition-transform duration-300">
                <div className="absolute inset-0.5 bg-dark rounded-sm">
                  <ChartBarSquareIcon className="absolute inset-1 text-accent" />
                </div>
              </div>
              <div className="flex flex-col">
                <span className="text-accent font-bold text-xl tracking-tight">Galcast.co</span>
                <span className="text-[10px] text-gray-400 -mt-1 tracking-wider">FINANCE DEMYSTIFIED</span>
              </div>
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="hidden md:flex items-center space-x-6"
          >
            <Link 
              to="/posts" 
              className="bg-accent text-black font-semibold px-6 py-2 rounded-lg hover:brightness-110 transition-all shadow-neon-gain"
            >
              Blog Posts
            </Link>
            <Link 
              to="/news" 
              className="px-4 py-2 text-white hover:bg-accent/10 rounded-lg transition-all duration-300 font-medium"
            >
              Stock Sentiment & Data
            </Link>
            {user ? (
              <>
                <Link 
                  to="/create"
                  className="bg-accent text-black font-semibold px-6 py-2 rounded-lg hover:brightness-110 transition-all shadow-neon-gain"
                >
                  Write Post
                </Link>
                <div className="flex items-center space-x-4">
                  <Link
                    to="/profile"
                    className="px-4 py-2 text-white hover:bg-accent/10 rounded-lg transition-all duration-300 font-medium"
                  >
                    Profile
                  </Link>
                  <button
                    onClick={handleSignOut}
                    className="px-4 py-2 text-white hover:bg-accent/10 rounded-lg transition-all duration-300 font-medium"
                  >
                    Sign Out
                  </button>
                </div>
              </>
            ) : (
              <Link 
                to="/login"
                className="px-4 py-2 text-white hover:bg-accent/10 rounded-lg transition-all duration-300 font-medium"
              >
                Sign In to Write
              </Link>
            )}
          </motion.div>
        </div>
      </div>
    </nav>
  );
};

export default TopNav;