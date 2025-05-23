import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Landing from './components/Landing';
import BlogPosts from './components/BlogPosts';
import BlogPost from './components/BlogPost';
import CreatePost from './components/CreatePost';
import EditPost from './components/EditPost';
import UserProfile from './components/UserProfile';
import Auth from './components/Auth';
import TopNav from './components/TopNav';
import Footer from './components/Footer';
import NewsAndSentiment from './components/NewsAndSentiment';
import PrivacyPolicy from './components/PrivacyPolicy';
import TermsOfService from './components/TermsOfService';
import DisclaimerPage from './components/DisclaimerPage';

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-dark">
        <TopNav />
        <div className="pt-16">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/posts" element={<BlogPosts />} />
            <Route path="/login" element={<Auth mode="login" />} />
            <Route path="/signup" element={<Auth mode="signup" />} />
            <Route path="/post/:id" element={<BlogPost />} />
            <Route path="/post/:id/edit" element={<EditPost />} />
            <Route path="/create" element={<CreatePost />} />
            <Route path="/profile" element={<UserProfile />} />
            <Route path="/news" element={<NewsAndSentiment />} />
            <Route path="/privacy" element={<PrivacyPolicy />} />
            <Route path="/terms" element={<TermsOfService />} />
            <Route path="/disclaimer" element={<DisclaimerPage />} />
          </Routes>
        </div>
        <Footer />
        <Toaster position="bottom-right" />
      </div>
    </Router>
  );
};

export default App;