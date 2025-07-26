"use client";

import React from "react";
import { motion } from "framer-motion";

interface PasswordStrengthProps {
  password: string;
}

const PasswordStrength: React.FC<PasswordStrengthProps> = ({ password }) => {
  const calculateStrength = (password: string) => {
    let strength = 0;
    if (password.length > 5) strength += 1;
    if (password.length > 7) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[0-9]/.test(password)) strength += 1;
    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    return Math.min(strength, 5);
  };

  const strength = calculateStrength(password);
  const strengthText =
    ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"][strength] ||
    "Very Weak";

  const colors = [
    "bg-red-500",
    "bg-orange-500",
    "bg-yellow-500",
    "bg-lime-500",
    "bg-green-500",
  ];

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{
        opacity: password ? 1 : 0,
        height: password ? "auto" : 0,
      }}
      className="mb-4"
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gray-700">
          Password strength
        </span>
      </div>
      <div className="flex space-x-1">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            className={`h-1.5 flex-1 rounded-full ${
              i < strength ? `${colors[strength - 1]}` : "bg-gray-200"
            }`}
          />
        ))}
      </div>
    </motion.div>
  );
};

export default PasswordStrength;
