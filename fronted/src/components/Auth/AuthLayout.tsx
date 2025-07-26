"use client";
import React, { ReactNode } from "react";
import Link from "next/link";
import { FiArrowLeft } from "react-icons/fi";
import { motion } from "framer-motion";

interface AuthLayoutProps {
  children: ReactNode;
  title?: string; // Add this line
}

const AuthLayout: React.FC<AuthLayoutProps> = ({ children, title }) => {
  return (
    <div className="min-h-screen flex flex-col md:flex-row">
      {/* Gradient Sidebar */}
      <motion.div
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="md:w-1/2 bg-gradient-to-br from-indigo-600 to-purple-700 flex flex-col justify-between p-8 md:p-12 text-white"
      >
        <div>
          <Link
            href="/"
            className="flex items-center text-white/80 hover:text-white transition-colors mb-16"
          >
            <FiArrowLeft className="mr-2" /> Back to home
          </Link>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="bg-white/10 backdrop-blur-sm p-6 rounded-2xl border border-white/20 max-w-md">
              <h1 className="text-3xl md:text-4xl font-bold mb-4">
                Welcome back!
              </h1>
              <p className="text-lg opacity-90">
                Sign in to access your personalized dashboard and continue your
                journey with us.
              </p>
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="hidden md:block"
        >
          <div className="flex space-x-4">
            {[...Array(3)].map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  y: [0, -15, 0],
                }}
                transition={{
                  duration: 2.5,
                  repeat: Infinity,
                  delay: i * 0.3,
                  ease: "easeInOut",
                }}
                className="bg-white/20 p-4 rounded-xl backdrop-blur-sm border border-white/30"
              >
                <div className="bg-gray-200 border-2 border-dashed rounded-xl w-16 h-16" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </motion.div>

      {/* Form Section */}
      <div className="md:w-1/2 flex items-center justify-center p-6 md:p-12 bg-gray-50">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-full max-w-md"
        >
          {children}
        </motion.div>
      </div>
    </div>
  );
};

export default AuthLayout;
