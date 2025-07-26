"use client";

import React, { useState } from "react";
import {
  FiLock,
  FiGithub,
  FiTwitter,
  FiFacebook,
  FiUser,
} from "react-icons/fi";
import { Formik, Form } from "formik";
import * as Yup from "yup";
import { toast } from "react-hot-toast";
import AuthLayout from "./AuthLayout";
import AuthInput from "@/components/Auth/AuthInput";
import { loginUser } from "@/store/slices/authSlice";
import { useAppDispatch } from "@/store";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";

interface LoginFormValues {
  username: string;
  password: string;
}
const SignIn = () => {
  const [loading, setLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const dispatch = useAppDispatch();

  const router = useRouter();

  const validationSchema = Yup.object({
    username: Yup.string().required("Username is required"),
    password: Yup.string().required("Password is required"),
  });

  const handleSubmit = async (values: LoginFormValues) => {
    setLoading(true);
    try {
      const result = await dispatch(loginUser(values));

      if (loginUser.fulfilled.match(result)) {
        const data = result.payload;
        if (data && data.access_token) {
          toast.success("Login successful!");
          router.push("/dashboard");
        }
      } else if (loginUser.rejected.match(result)) {
        // Handle specific error cases
        const errorPayload = result.payload as {
          message?: string;
          code?: number;
        };

        if (errorPayload?.message === "Bad credentials") {
          toast.error("Invalid username or password");
        } else if (errorPayload?.message) {
          toast.error(errorPayload.message);
        } else {
          toast.error("Login failed. Please try again.");
        }
      }
    } catch (err) {
      toast.error("Unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 },
    },
  };

  return (
    <AuthLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="text-center mb-8"
      >
        <div className="flex justify-center mb-4">
          <div className="bg-indigo-100 p-3 rounded-full">
            <div className="bg-indigo-600 text-white p-3 rounded-full">
              <FiLock size={28} />
            </div>
          </div>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Sign in to your account
        </h2>
        <p className="text-gray-600">
          Enter your credentials to access your account
        </p>
      </motion.div>

      <Formik
        initialValues={{ username: "", password: "" }}
        validationSchema={validationSchema}
        onSubmit={handleSubmit}
      >
        {({ values, errors, touched, handleChange }) => (
          <Form>
            <AuthInput
              icon={<FiUser />}
              name="username"
              type="text"
              placeholder="Username or Email"
              value={values.username}
              onChange={handleChange}
              error={touched.username && errors.username}
            />

            <AuthInput
              icon={<FiLock />}
              name="password"
              type="password"
              placeholder="Password"
              value={values.password}
              onChange={handleChange}
              error={touched.password && errors.password}
            />

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="flex items-center justify-between mb-6"
            >
              <label className="flex items-center cursor-pointer">
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={() => setRememberMe(!rememberMe)}
                    className="sr-only"
                  />
                  <div
                    className={`block w-5 h-5 rounded border ${
                      rememberMe
                        ? "bg-indigo-600 border-indigo-600"
                        : "bg-white border-gray-400"
                    }`}
                  >
                    {rememberMe && (
                      <svg
                        className="w-4 h-4 mx-auto text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="3"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </div>
                </div>
                <span className="ml-2 text-sm text-gray-700">Remember me</span>
              </label>

              <Link
                href="/forgetpassword"
                className="text-sm font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
              >
                Forgot password?
              </Link>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className="mb-6"
            >
              <button
                type="submit"
                disabled={loading}
                className={`w-full bg-gradient-to-r from-indigo-600 to-purple-700 text-white font-semibold py-3.5 px-4 rounded-xl transition-all duration-300 shadow-lg hover:shadow-xl ${
                  loading
                    ? "opacity-80 cursor-not-allowed"
                    : "hover:from-indigo-700 hover:to-purple-800"
                }`}
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Signing in...
                  </div>
                ) : (
                  "Sign in"
                )}
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="relative my-8"
            >
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300"></div>
              </div>
              <div className="relative flex justify-center">
                <span className="px-3 bg-gray-50 text-gray-500 text-sm">
                  Or continue with
                </span>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="flex justify-center space-x-4 mb-6"
            >
              <button
                type="button"
                className="p-3 border border-gray-300 rounded-xl hover:bg-gray-100 transition-colors duration-300 hover:border-indigo-300"
              >
                <FiGithub className="text-gray-700" size={20} />
              </button>
              <button
                type="button"
                className="p-3 border border-gray-300 rounded-xl hover:bg-blue-50 transition-colors duration-300 hover:border-blue-300"
              >
                <FiTwitter className="text-blue-400" size={20} />
              </button>
              <button
                type="button"
                className="p-3 border border-gray-300 rounded-xl hover:bg-blue-50 transition-colors duration-300 hover:border-blue-300"
              >
                <FiFacebook className="text-blue-600" size={20} />
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="text-center text-gray-600"
            >
              <p>
                Don't have an account?{" "}
                <Link
                  href="/signup"
                  className="font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
                >
                  Sign up
                </Link>
              </p>
            </motion.div>
          </Form>
        )}
      </Formik>
    </AuthLayout>
  );
};

export default SignIn;
