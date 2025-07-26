// Header.tsx
"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { LogOut, Settings, Menu, X } from "lucide-react";
import { useAppDispatch } from "@/store";
import { logoutUser } from "@/store/slices/authSlice";
import Button from "@/components/UI/Button";
import { motion, AnimatePresence } from "framer-motion";

const Header = () => {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const handleLogout = async () => {
    await dispatch(logoutUser());
    router.push("/signin");
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-indigo-600 to-purple-600 shadow-lg">
      <div className="max-w-7xl mx-52 px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-16">
          <div className="flex items-center">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 300 }}
              className="flex-shrink-0"
            >
              <div className="bg-white/20 p-2 rounded-lg backdrop-blur-sm">
                <div className="w-8 h-8 bg-gradient-to-tr from-amber-400 to-red-500 rounded-md" />
              </div>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="ml-4 text-xl font-bold text-white tracking-tight"
            >
              <span className="text-amber-300">NexaSpeak</span> Institute
            </motion.h1>
          </div>
          <div className="ml-auto flex items-center">
            <div className="hidden md:flex items-center space-x-2">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => router.push("/profile")}
                  className="text-white hover:bg-white/20 rounded-full p-2 transition-all duration-300"
                >
                  <Settings className="w-5 h-5" />
                </Button>
              </motion.div>

              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  className="text-white hover:bg-white/20 rounded-full p-2 transition-all duration-300"
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                >
                  <LogOut className="w-5 h-5" />
                </Button>
              </motion.div>
            </div>

            <div className="md:hidden">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowMobileMenu(!showMobileMenu)}
                className="text-white p-2"
              >
                {showMobileMenu ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {showMobileMenu && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden bg-indigo-700"
          >
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 flex flex-col items-end">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => router.push("/profile")}
                  className="text-white hover:bg-white/20 rounded-full p-2 my-1 transition-all duration-300"
                >
                  <Settings className="w-5 h-5 mr-2" /> Settings
                </Button>
              </motion.div>

              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  className="text-white hover:bg-white/20 rounded-full p-2 my-1 transition-all duration-300"
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                >
                  <LogOut className="w-5 h-5 mr-2" /> Logout
                </Button>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

export default Header;
