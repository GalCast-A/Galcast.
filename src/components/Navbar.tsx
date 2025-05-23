import React, { useState, useEffect } from 'react';

const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${
      scrolled ? 'bg-darker/90 backdrop-blur-md shadow-lg' : 'bg-transparent'
    }`}>
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center space-x-3 group cursor-pointer">
            <div className="w-8 h-8 bg-gain rounded-lg transform transition-transform group-hover:rotate-12"></div>
            <span className="text-gain font-bold text-2xl tracking-tight">GALCAST P&O</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <NavLink href="#features">Features</NavLink>
            <NavLink href="#metrics">Metrics</NavLink>
            <NavLink href="#clients">Clients</NavLink>
            <NavLink href="#about">About</NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
};

const NavLink: React.FC<{ href: string; children: React.ReactNode }> = ({ href, children }) => (
  <a 
    href={href} 
    className="text-gray-400 hover:text-gain transition-colors relative group"
  >
    {children}
    <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gain transition-all duration-300 group-hover:w-full"></span>
  </a>
);

export default Navbar;