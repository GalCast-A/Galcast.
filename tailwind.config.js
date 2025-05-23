/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: '#00B050', // Darker, more professional stock market green
        gain: '#007A35', // Darker secondary green for gains
        loss: '#CC2929', // Darker red for losses
        dark: '#0A0A0A', // Darker background
        card: '#141414', // Darker card background
        border: 'rgba(255, 255, 255, 0.05)', // More subtle borders
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        card: '0 4px 6px rgba(0, 0, 0, 0.2)',
        'card-hover': '0 6px 12px rgba(0, 0, 0, 0.25)',
        'neon-gain': '0 0 20px rgba(0, 176, 80, 0.2)',
        'neon-loss': '0 0 20px rgba(204, 41, 41, 0.2)',
      },
      animation: {
        'pulse-gain': 'pulse-gain 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        'pulse-gain': {
          '0%, 100%': {
            opacity: 1,
            boxShadow: '0 0 20px rgba(0, 176, 80, 0.2)',
          },
          '50%': {
            opacity: 0.8,
            boxShadow: '0 0 30px rgba(0, 176, 80, 0.3)',
          },
        },
      },
    },
  },
  plugins: [],
}