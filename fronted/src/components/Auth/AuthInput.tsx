"use client";

import React, { useState } from "react";
import {
  FiLock,
  FiGithub,
  FiTwitter,
  FiFacebook,
  FiUser,
  FiEye,
  FiEyeOff,
  FiArrowLeft,
} from "react-icons/fi";
import { motion } from "framer-motion";
import { Formik, Form } from "formik";
import * as Yup from "yup";
import { toast } from "react-hot-toast";
import { loginUser } from "@/store/slices/authSlice";
import { useAppDispatch } from "@/store";
import { useRouter } from "next/navigation";
import Link from "next/link";

interface LoginFormValues {
  username: string;
  password: string;
}

const AuthInput = ({
  icon,
  type = "text",
  placeholder,
  value,
  onChange,
  error,
  ...props
}) => {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="mb-6"
    >
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-indigo-600">
          {icon}
        </div>
        <input
          className={`text-black w-full pl-10 pr-10 py-3.5 border ${
            error ? "border-rose-500" : "border-gray-300"
          } rounded-xl bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-300 shadow-sm`}
          type={
            type === "password" ? (showPassword ? "text" : "password") : type
          }
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          {...props}
        />
        {type === "password" && (
          <button
            type="button"
            className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-indigo-700 transition-colors"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? <FiEyeOff size={20} /> : <FiEye size={20} />}
          </button>
        )}
      </div>
      {error && (
        <motion.p
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="text-rose-500 text-sm mt-1.5 ml-1"
        >
          {error}
        </motion.p>
      )}
    </motion.div>
  );
};

export default AuthInput;
