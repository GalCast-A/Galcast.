import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
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
  author_id: string;
}

export default function BlogsAndPosts() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPosts();
  }, []);

  async function fetchPosts() {
    try {
      const { data, error } = await supabase
        .from('posts')
        .select('*')
        .eq('published', true)
        .order('created_at', { ascending: false });

      if (error) {
        throw error;
      }

      setPosts(data || []);
    } catch (error) {
      console.error('Error fetching posts:', error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="bg-gray-900 min-h-screen flex items-center justify-center">
        <div className="text-white">Loading posts...</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto max-w-2xl text-center"
        >
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Latest Blog Posts
          </h2>
          <p className="mt-2 text-lg leading-8 text-gray-300">
            Explore our collection of articles and insights
          </p>
          <div className="mt-4">
            <Link
              to="/blogs/create"
              className="inline-block rounded-md bg-indigo-600 px-4 py-2 text-base font-semibold text-white shadow-sm hover:bg-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900"
            >
              Create New Post
            </Link>
          </div>
        </motion.div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-x-8 gap-y-20 lg:mx-0 lg:max-w-none lg:grid-cols-3">
          {posts.map((post) => (
            <motion.article
              key={post.id}
              className="flex flex-col items-start bg-gray-800 rounded-lg overflow-hidden"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.2 }}
            >
              {post.featured_image && (
                <div className="relative w-full h-48">
                  <img
                    src={post.featured_image}
                    alt={post.title}
                    className="w-full h-full object-cover"
                  />
                </div>
              )}
              <div className="p-6 flex flex-col flex-1">
                <div className="flex items-center gap-x-4 text-xs mb-4">
                  <time dateTime={post.created_at} className="text-gray-400">
                    {format(new Date(post.created_at), 'MMMM d, yyyy')}
                  </time>
                  <span className="relative z-10 rounded-full bg-indigo-600 px-3 py-1.5 text-white">
                    {post.category}
                  </span>
                </div>
                <div className="group relative flex-1">
                  <h3 className="text-lg font-semibold leading-6 text-white group-hover:text-gray-300">
                    <Link to={`/blogs/${post.id}`}>
                      <span className="absolute inset-0" />
                      {post.title}
                    </Link>
                  </h3>
                  <p className="mt-5 line-clamp-3 text-sm leading-6 text-gray-300">
                    {post.content}
                  </p>
                </div>
                <div className="mt-4 flex items-center gap-x-4">
                  <div className="text-sm leading-6">
                    <p className="font-semibold text-white">
                      <Link to={`/blogs/${post.id}`} className="hover:underline">
                        Read more →
                      </Link>
                    </p>
                  </div>
                </div>
              </div>
            </motion.article>
          ))}
        </div>

        {posts.length === 0 && (
          <div className="text-center mt-16">
            <p className="text-gray-300">No posts found. Be the first to create one!</p>
          </div>
        )}
      </div>
    </div>
  );
}