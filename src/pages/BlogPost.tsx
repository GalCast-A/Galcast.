import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { supabase } from '../supabaseClient';
import { format } from 'date-fns';

interface Post {
  id: string;
  title: string;
  content: string;
  created_at: string;
  category: string;
  featured_image: string;
  about_author: string;
  sources: string[];
  author_id: string;
}

export default function BlogPost() {
  const { id } = useParams<{ id: string }>();
  const [post, setPost] = useState<Post | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPost();
  }, [id]);

  async function fetchPost() {
    if (!id) return;

    try {
      const { data, error } = await supabase
        .from('posts')
        .select('*')
        .eq('id', id)
        .single();

      if (error) throw error;
      setPost(data);
    } catch (err) {
      console.error('Error fetching post:', err);
      setError(err instanceof Error ? err.message : 'Error fetching post');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="bg-gray-900 min-h-screen flex items-center justify-center">
        <div className="text-white">Loading post...</div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="bg-gray-900 min-h-screen py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl">
            <div className="text-center text-white">
              <h2 className="text-2xl font-bold">Error</h2>
              <p className="mt-4 text-gray-300">{error || 'Post not found'}</p>
              <Link
                to="/blogs"
                className="mt-8 inline-block text-indigo-400 hover:text-indigo-300"
              >
                ← Back to Posts
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <motion.article
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto max-w-2xl"
        >
          <Link
            to="/blogs"
            className="text-indigo-400 hover:text-indigo-300 mb-8 inline-flex items-center gap-2"
          >
            <span>←</span> Back to Posts
          </Link>

          {post.featured_image && (
            <motion.img
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              src={post.featured_image}
              alt={post.title}
              className="w-full h-[400px] object-cover rounded-lg mb-8 shadow-xl"
            />
          )}

          <div className="flex items-center gap-x-4 text-xs mb-4">
            <time dateTime={post.created_at} className="text-gray-400">
              {format(new Date(post.created_at), 'MMMM d, yyyy')}
            </time>
            <span className="relative z-10 rounded-full bg-indigo-600 px-3 py-1.5 text-white font-medium">
              {post.category}
            </span>
          </div>

          <h1 className="text-4xl font-bold tracking-tight text-white mb-8">
            {post.title}
          </h1>

          <div className="prose prose-lg prose-invert max-w-none">
            <div className="text-gray-300 whitespace-pre-wrap leading-relaxed">
              {post.content}
            </div>
          </div>

          {post.about_author && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="mt-12 border-t border-gray-700 pt-8"
            >
              <h2 className="text-2xl font-bold text-white mb-4">About the Author</h2>
              <p className="text-gray-300 leading-relaxed">{post.about_author}</p>
            </motion.div>
          )}

          {post.sources && post.sources.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="mt-12 border-t border-gray-700 pt-8"
            >
              <h2 className="text-2xl font-bold text-white mb-4">Sources</h2>
              <ul className="space-y-2 text-gray-300">
                {post.sources.map((source, index) => (
                  <li key={index}>
                    <a
                      href={source}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-indigo-400 hover:text-indigo-300 break-all"
                    >
                      {source}
                    </a>
                  </li>
                ))}
              </ul>
            </motion.div>
          )}
        </motion.article>
      </div>
    </div>
  );
}