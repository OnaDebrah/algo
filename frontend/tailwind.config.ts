/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#667eea',
          50: '#f5f7ff',
          100: '#ebedff',
          200: '#d6dcff',
          300: '#b8c1ff',
          400: '#8b9cff',
          500: '#667eea',
          600: '#5568d3',
          700: '#4451b2',
          800: '#363f91',
          900: '#2d3478',
        },
        secondary: {
          DEFAULT: '#764ba2',
          50: '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#764ba2',
          600: '#667eea',
          700: '#5a67d8',
          800: '#4c51bf',
          900: '#434190',
        },
        success: {
          DEFAULT: '#00c853',
          light: '#00e676',
          dark: '#00a843'
        },
        danger: {
          DEFAULT: '#ff1744',
          light: '#ff5252',
          dark: '#d50000'
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
      animation: {
        'fade-in': 'fade-in 0.5s ease-out',
        'slide-in': 'slide-in-left 0.3s ease-out',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-left': {
          '0%': { opacity: '0', transform: 'translateX(-20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' }
        },
        'pulse-glow': {
          '0%, 100%': { boxShadow: '0 0 20px rgba(102, 126, 234, 0.4)' },
          '50%': { boxShadow: '0 0 30px rgba(102, 126, 234, 0.6)' }
        }
      }
    }
  }
}
