"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiMail, FiUser, FiEdit2, FiCheck, FiLock, FiGlobe, FiBell, FiCreditCard, FiShield, FiLogOut } from "react-icons/fi";

const ProfilePage = () => {
  const [isEditing, setIsEditing] = useState(false);
  const [username, setUsername] = useState("testuser");
  const [email, setEmail] = useState("test@yopmail.com");
  const [tempUsername, setTempUsername] = useState(username);
  const [tempEmail, setTempEmail] = useState(email);
  const [activeTab, setActiveTab] = useState("profile");
  const [isLoading, setIsLoading] = useState(true);

  // Simulate loading effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1200);
    return () => clearTimeout(timer);
  }, []);

  const handleEditToggle = () => {
    if (isEditing) {
      setUsername(tempUsername);
      setEmail(tempEmail);
    }
    setIsEditing(!isEditing);
  };

  const handleSave = () => {
    setIsEditing(false);
    setUsername(tempUsername);
    setEmail(tempEmail);
  };

  const tabItems = [
    { id: "profile", icon: <FiUser className="mr-2" />, label: "Profile" },
    { id: "security", icon: <FiLock className="mr-2" />, label: "Security" },
    { id: "notifications", icon: <FiBell className="mr-2" />, label: "Notifications" },
    { id: "billing", icon: <FiCreditCard className="mr-2" />, label: "Billing" },
    { id: "preferences", icon: <FiGlobe className="mr-2" />, label: "Preferences" },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-600"></div>
          <p className="mt-4 text-lg font-medium text-gray-700">Loading your profile...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8"
    >
      <div className="mb-10">
        <motion.h1 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="text-3xl md:text-4xl font-bold text-gray-900"
        >
          Profile
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="mt-2 text-lg text-gray-600"
        >
          Manage your account settings and preferences
        </motion.p>
      </div>

      <div className="flex flex-col md:flex-row gap-6">
        {/* Sidebar */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="w-full md:w-64 flex-shrink-0"
        >
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
            <div className="p-6 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
              <div className="flex items-center">
                <div className="bg-gray-200 border-2 border-dashed rounded-xl w-16 h-16 flex items-center justify-center">
                  <FiUser className="text-3xl text-white" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-bold">{username}</h3>
                  <p className="text-indigo-100 text-sm">{email}</p>
                </div>
              </div>
            </div>
            
            <div className="py-4">
              {tabItems.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center w-full px-6 py-3 text-left transition-colors duration-200 ${
                    activeTab === tab.id
                      ? "bg-indigo-50 text-indigo-700 border-l-4 border-indigo-600"
                      : "text-gray-600 hover:bg-gray-50"
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>
            
            <button className="flex items-center w-full px-6 py-3 text-left text-gray-600 hover:bg-gray-50 transition-colors duration-200 mt-4">
              <FiLogOut className="mr-2" />
              Logout
            </button>
          </div>
        </motion.div>

        {/* Main Content */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="flex-grow"
        >
          {activeTab === "profile" && (
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
              <div className="px-6 py-5 border-b border-gray-100 flex justify-between items-center">
                <h2 className="text-xl font-bold text-gray-900">User Information</h2>
                {isEditing ? (
                  <button
                    onClick={handleSave}
                    className="flex items-center px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-green-300"
                  >
                    <FiCheck className="mr-2" />
                    Save Changes
                  </button>
                ) : (
                  <button
                    onClick={handleEditToggle}
                    className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-indigo-300"
                  >
                    <FiEdit2 className="mr-2" />
                    Edit Profile
                  </button>
                )}
              </div>
              
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-50 rounded-xl p-5 border border-gray-100 transition-all duration-300 hover:shadow-md">
                    <div className="flex items-center mb-3">
                      <div className="bg-indigo-100 p-2 rounded-lg">
                        <FiMail className="text-indigo-600 text-xl" />
                      </div>
                      <h3 className="ml-3 font-medium text-gray-700">Email</h3>
                    </div>
                    {isEditing ? (
                      <input
                        type="email"
                        value={tempEmail}
                        onChange={(e) => setTempEmail(e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 outline-none transition-all"
                      />
                    ) : (
                      <p className="text-gray-900 font-medium">{email}</p>
                    )}
                  </div>
                  
                  <div className="bg-gray-50 rounded-xl p-5 border border-gray-100 transition-all duration-300 hover:shadow-md">
                    <div className="flex items-center mb-3">
                      <div className="bg-indigo-100 p-2 rounded-lg">
                        <FiUser className="text-indigo-600 text-xl" />
                      </div>
                      <h3 className="ml-3 font-medium text-gray-700">Username</h3>
                    </div>
                    {isEditing ? (
                      <input
                        type="text"
                        value={tempUsername}
                        onChange={(e) => setTempUsername(e.target.value)}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 outline-none transition-all"
                      />
                    ) : (
                      <p className="text-gray-900 font-medium">{username}</p>
                    )}
                  </div>
                </div>
                
                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Security</h3>
                  <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-5 border border-indigo-100">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <FiShield className="text-indigo-600 text-2xl mr-3" />
                        <div>
                          <h4 className="font-medium text-gray-900">Two-Factor Authentication</h4>
                          <p className="text-sm text-gray-600">Add an extra layer of security</p>
                        </div>
                      </div>
                      <button className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors duration-300">
                        Enable
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === "security" && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Security Settings</h2>
              <div className="space-y-5">
                <div className="p-5 bg-gray-50 rounded-xl border border-gray-100 hover:shadow-md transition-all">
                  <div className="flex justify-between items-center">
                    <h3 className="font-medium text-gray-900">Password</h3>
                    <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors">
                      Change
                    </button>
                  </div>
                  <p className="text-gray-600 mt-2">Last changed 3 months ago</p>
                </div>
                
                <div className="p-5 bg-gray-50 rounded-xl border border-gray-100 hover:shadow-md transition-all">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-medium text-gray-900">Two-Factor Authentication</h3>
                      <p className="text-gray-600 mt-1">Add an extra layer of security</p>
                    </div>
                    <div className="relative inline-block w-12 h-6">
                      <input type="checkbox" className="opacity-0 w-0 h-0 peer" />
                      <div className="absolute cursor-pointer top-0 left-0 right-0 bottom-0 bg-gray-300 rounded-full transition peer-checked:bg-indigo-600 before:absolute before:h-4 before:w-4 before:left-1 before:bottom-1 before:bg-white before:rounded-full before:transition peer-checked:before:translate-x-6"></div>
                    </div>
                  </div>
                </div>
                
                <div className="p-5 bg-gray-50 rounded-xl border border-gray-100 hover:shadow-md transition-all">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-medium text-gray-900">Active Sessions</h3>
                      <p className="text-gray-600 mt-1">2 active sessions</p>
                    </div>
                    <button className="px-4 py-2 text-indigo-600 hover:text-indigo-800 transition-colors">
                      View all
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
};

export default ProfilePage;