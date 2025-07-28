// Layout.tsx
"use client";

import React, { useEffect, useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import { motion } from "framer-motion";
import { usePathname } from "next/navigation";

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const pathname = usePathname();
  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Header setSidebarOpen={setSidebarOpen} />
      <div className="flex pt-16">
        <Sidebar
          collapsed={sidebarCollapsed}
          setCollapsed={setSidebarCollapsed}
          isOpen={sidebarOpen}
          setIsOpen={setSidebarOpen}
        />
        <motion.main
          className={`flex-1 min-h-[calc(100vh-4rem)] p-4 md:p-8 transition-all duration-300 ${
            sidebarCollapsed ? "ml-20" : "ml-64"
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="max-w-7xl mx-auto">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="bg-white rounded-2xl shadow-xl p-6 overflow-hidden"
            >
              {children}
            </motion.div>

            <motion.footer
              className="mt-8 text-center text-gray-500 text-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              © {new Date().getFullYear()} Digital Archival System. All rights
              reserved.
            </motion.footer>
          </div>
        </motion.main>
      </div>
    </div>
  );
};

export default Layout;
