// app/forgotten-password/[token]/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import { FiLock, FiCheckCircle } from "react-icons/fi";
import { Formik, Form } from "formik";
import * as Yup from "yup";
import { toast } from "react-hot-toast";
import AuthLayout from "@/app/(authenticated)/layout";
import AuthInput from "@/components/Auth/AuthInput";
import Link from "next/link";
import { motion } from "framer-motion";
import { useSearchParams } from "next/navigation";

const ResetPassword = () => {
  const [loading, setLoading] = useState(false);
  const [validToken, setValidToken] = useState(false);
  const [email, setEmail] = useState("");
  const [success, setSuccess] = useState(false);
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  useEffect(() => {
    const validateToken = async () => {
      if (!token) return;

      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/validate-reset-token/${token}`
        );

        if (response.ok) {
          const data = await response.json();
          setValidToken(true);
          setEmail(data.email);
        } else {
          toast.error("Invalid or expired token");
          setValidToken(false);
        }
      } catch (error) {
        toast.error("Failed to validate token");
      }
    };

    validateToken();
  }, [token]);

  const validationSchema = Yup.object({
    password: Yup.string()
      .min(8, "Password must be at least 8 characters")
      .required("Password is required"),
    confirmPassword: Yup.string()
      .oneOf([Yup.ref("password")], "Passwords must match")
      .required("Confirm password is required"),
  });

  const handleSubmit = async (values: {
    password: string;
    confirmPassword: string;
  }) => {
    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/reset-password`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            token,
            new_password: values.password,
            confirm_password: values.confirmPassword,
          }),
        }
      );

      if (response.ok) {
        toast.success("Password has been reset successfully");
        setSuccess(true);
      } else {
        const errorData = await response.json();
        toast.error(errorData.detail || "Failed to reset password");
      }
    } catch (error) {
      toast.error("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <AuthLayout>
        <div className="text-center p-8">
          <h2 className="text-2xl font-bold text-red-600 mb-4">
            Invalid Request
          </h2>
          <p className="text-gray-600 mb-6">
            Missing reset token. Please check your email for the correct link.
          </p>
          <Link
            href="/forgetpassword"
            className="text-indigo-600 hover:text-indigo-800 font-medium"
          >
            Request a new reset link
          </Link>
        </div>
      </AuthLayout>
    );
  }

  if (!validToken) {
    return (
      <AuthLayout>
        <div className="text-center p-8">
          <div className="flex justify-center mb-4">
            <div className="bg-red-100 p-3 rounded-full">
              <div className="bg-red-600 text-white p-3 rounded-full">
                <FiLock size={28} />
              </div>
            </div>
          </div>
          <h2 className="text-2xl font-bold text-red-600 mb-4">
            Invalid Token
          </h2>
          <p className="text-gray-600 mb-6">
            This password reset link is invalid or has expired.
          </p>
          <Link
            href="/forgotten-password"
            className="text-indigo-600 hover:text-indigo-800 font-medium"
          >
            Request a new reset link
          </Link>
        </div>
      </AuthLayout>
    );
  }

  if (success) {
    return (
      <AuthLayout>
        <div className="text-center p-8">
          <div className="flex justify-center mb-4">
            <div className="bg-green-100 p-3 rounded-full">
              <div className="bg-green-600 text-white p-3 rounded-full">
                <FiCheckCircle size={28} />
              </div>
            </div>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Password Reset Successfully
          </h2>
          <p className="text-gray-600 mb-6">
            Your password has been updated successfully.
          </p>
          <Link
            href="/signin"
            className="inline-flex items-center justify-center w-full bg-indigo-600 text-white font-semibold py-3 px-4 rounded-xl hover:bg-indigo-700 transition-colors"
          >
            Sign In Now
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
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
          Reset Your Password
        </h2>
        <p className="text-gray-600">
          Enter a new password for <span className="font-medium">{email}</span>
        </p>
      </motion.div>

      <Formik
        initialValues={{ password: "", confirmPassword: "" }}
        validationSchema={validationSchema}
        onSubmit={handleSubmit}
      >
        {({ values, errors, touched, handleChange }) => (
          <Form>
            <AuthInput
              icon={<FiLock />}
              name="password"
              type="password"
              placeholder="New Password"
              value={values.password}
              onChange={handleChange}
              error={touched.password && errors.password}
            />

            <AuthInput
              icon={<FiLock />}
              name="confirmPassword"
              type="password"
              placeholder="Confirm New Password"
              value={values.confirmPassword}
              onChange={handleChange}
              error={touched.confirmPassword && errors.confirmPassword}
            />

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
                    Resetting...
                  </div>
                ) : (
                  "Reset Password"
                )}
              </button>
            </motion.div>
          </Form>
        )}
      </Formik>
    </AuthLayout>
  );
};

export default ResetPassword;
