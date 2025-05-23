import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="mx-auto max-w-7xl px-6 py-12 md:flex md:items-center md:justify-between lg:px-8">
        <div className="mt-8 md:order-1 md:mt-0">
          <nav className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-6">
            <Link to="/disclaimer" className="text-gray-300 hover:text-white">
              Disclaimer
            </Link>
            <Link to="/privacy-policy" className="text-gray-300 hover:text-white">
              Privacy Policy
            </Link>
            <Link to="/terms-of-service" className="text-gray-300 hover:text-white">
              Terms of Service
            </Link>
          </nav>
        </div>
        <div className="mt-8 md:order-2 md:mt-0">
          <p className="text-center text-xs leading-5 text-gray-400">
            &copy; {new Date().getFullYear()} FinanceApp. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}