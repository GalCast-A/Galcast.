import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { supabase } from '../supabaseClient';

export default function CreatePost() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    category: '',
    featured_image: '',
    about_author: '',
    sources: [] as string[],
  });
  const [currentSource, setCurrentSource] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const { data: userData, error: userError } = await supabase.auth.getUser();
      if (userError || !userData.user) {
        throw new Error('Please sign in to create a post');
      }

      const { data, error } = await supabase
        .from('posts')
        .insert([
          {
            ...formData,
            author_id: userData.user.id,
            sources: formData.sources.length > 0 ? formData.sources : null,
            published: true,
          },
        ])
        .select();

      if (error) throw error;

      navigate(`/blogs/${data[0].id}`);
    } catch (error) {
      console.error('Error creating post:', error);
      alert(error instanceof Error ? error.message : 'Error creating post');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const addSource = () => {
    if (currentSource) {
      setFormData({
        ...formData,
        sources: [...formData.sources, currentSource],
      });
      setCurrentSource('');
    }
  };

  const removeSource = (index: number) => {
    setFormData({
      ...formData,
      sources: formData.sources.filter((_, i) => i !== index),
    });
  };

  return (
    <div className="bg-gray-900 py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto max-w-2xl"
        >
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl mb-8">
            Create New Post
          </h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="title" className="block text-sm font-medium text-white">
                Title
              </label>
              <input
                type="text"
                name="title"
                id="title"
                required
                minLength={3}
                className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={formData.title}
                onChange={handleChange}
              />
            </div>

            <div>
              <label htmlFor="category" className="block text-sm font-medium text-white">
                Category
              </label>
              <select
                name="category"
                id="category"
                required
                className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={formData.category}
                onChange={handleChange}
              >
                <option value="">Select a category</option>
                <option value="Technology">Technology</option>
                <option value="Science">Science</option>
                <option value="Programming">Programming</option>
                <option value="Web Development">Web Development</option>
                <option value="AI & ML">AI & ML</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div>
              <label htmlFor="featured_image" className="block text-sm font-medium text-white">
                Featured Image URL
              </label>
              <input
                type="url"
                name="featured_image"
                id="featured_image"
                placeholder="https://example.com/image.jpg"
                className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={formData.featured_image}
                onChange={handleChange}
              />
            </div>

            <div>
              <label htmlFor="content" className="block text-sm font-medium text-white">
                Content
              </label>
              <textarea
                name="content"
                id="content"
                rows={8}
                required
                className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={formData.content}
                onChange={handleChange}
              />
            </div>

            <div>
              <label htmlFor="about_author" className="block text-sm font-medium text-white">
                About the Author
              </label>
              <textarea
                name="about_author"
                id="about_author"
                rows={3}
                className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                value={formData.about_author}
                onChange={handleChange}
                placeholder="Tell readers about yourself..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-white">Sources</label>
              <div className="mt-1 flex gap-2">
                <input
                  type="url"
                  value={currentSource}
                  onChange={(e) => setCurrentSource(e.target.value)}
                  className="flex-1 rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                  placeholder="Enter source URL"
                />
                <button
                  type="button"
                  onClick={addSource}
                  className="rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900"
                >
                  Add
                </button>
              </div>
              {formData.sources.length > 0 && (
                <ul className="mt-2 space-y-2">
                  {formData.sources.map((source, index) => (
                    <li key={index} className="flex items-center gap-2 text-gray-300">
                      <span className="flex-1 truncate">{source}</span>
                      <button
                        type="button"
                        onClick={() => removeSource(index)}
                        className="text-red-500 hover:text-red-400"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="flex justify-end gap-x-4">
              <button
                type="button"
                onClick={() => navigate('/blogs')}
                className="rounded-md bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-gray-600 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-900"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting ? 'Creating...' : 'Create Post'}
              </button>
            </div>
          </form>
        </motion.div>
      </div>
    </div>
  );
}