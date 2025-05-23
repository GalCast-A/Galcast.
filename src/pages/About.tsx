import React from 'react';

export default function About() {
  return (
    <div className="bg-white py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">About Us</h2>
          <div className="mt-6 space-y-6 text-base leading-7 text-gray-600">
            <p>
              Welcome to FinanceApp, your trusted source for financial education and insights. Our mission is to make financial literacy accessible to everyone, helping you make informed decisions about your money and investments.
            </p>
            <p>
              Founded with the belief that everyone deserves to understand finance, we provide clear, concise, and practical information about various financial topics. From basic concepts to advanced investment strategies, we're here to guide you on your financial journey.
            </p>
            <p>
              Our team consists of experienced financial professionals and educators who are passionate about sharing their knowledge and helping others achieve their financial goals.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}