// Sidebar.tsx
"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  User,
  UserPlus,
  MessageCircle,
  Mic,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";
import { motion, AnimatePresence } from "framer-motion";

interface SidebarProps {
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

const Sidebar = ({ collapsed, setCollapsed }: SidebarProps) => {
  const pathname = usePathname();

  const navigationItems = [
    {
      name: "Student Form",
      href: "/dashboard",
      icon: User,
    },
    {
      name: "Teacher Form",
      href: "/teacher",
      icon: UserPlus,
    },
    {
      name: "Chat with us",
      href: "/chat",
      icon: MessageCircle,
    },
    {
      name: "Speech Practice",
      href: "/voiceassistant",
      icon: Mic,
    },
  ];

  return (
    <motion.aside
      className={`fixed top-16 bottom-0 z-40 bg-gradient-to-b from-gray-900 to-gray-800 shadow-xl ${
        collapsed ? "w-20" : "w-64"
      }`}
      initial={{ width: 256 }}
      animate={{ width: collapsed ? 80 : 256 }}
      transition={{ type: "spring", damping: 20 }}
    >
      <div className="flex flex-col h-full">
        <div className="p-4 flex justify-end">
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setCollapsed(!collapsed)}
            className="p-2 rounded-full bg-gray-700 text-gray-300 hover:bg-indigo-600 hover:text-white transition-colors"
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </motion.button>
        </div>

        <nav className="p-4 space-y-1 flex-1">
          {navigationItems.map((item, index) => {
            const isActive = pathname === item.href;
            return (
              <motion.div
                key={item.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link href={item.href}>
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={clsx(
                      "flex items-center p-3 rounded-xl mb-2 transition-all duration-300 group",
                      isActive
                        ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg"
                        : "text-gray-300 hover:bg-gray-700"
                    )}
                  >
                    <item.icon className="w-6 h-6 flex-shrink-0" />
                    <AnimatePresence>
                      {!collapsed && (
                        <motion.span
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -10 }}
                          transition={{ duration: 0.2 }}
                          className={clsx(
                            "ml-3",
                            collapsed ? "hidden" : "block"
                          )}
                        >
                          {item.name}
                        </motion.span>
                      )}
                    </AnimatePresence>
                  </motion.div>
                </Link>
              </motion.div>
            );
          })}
        </nav>

        <motion.div
          className="p-4 border-t border-gray-700"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex items-center">
            <div className="bg-gradient-to-tr from-cyan-400 to-blue-500 w-10 h-10 rounded-full" />
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="ml-3"
                >
                  <p className="text-white text-sm font-medium">John Doe</p>
                  <p className="text-gray-400 text-xs">Administrator</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </motion.aside>
  );
};

export default Sidebar;
