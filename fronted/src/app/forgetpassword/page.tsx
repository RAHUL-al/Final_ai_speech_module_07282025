"use client";

import React, { useState } from "react";
import { FiMail, FiArrowLeft } from "react-icons/fi";
import { Formik, Form } from "formik";
import * as Yup from "yup";
import { toast } from "react-hot-toast";
import AuthLayout from "../(authenticated)/layout";
import AuthInput from "@/components/Auth/AuthInput";
import Link from "next/link";
import { motion } from "framer-motion";

const ForgotPassword = () => {
  const [loading, setLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  const validationSchema = Yup.object({
    email: Yup.string()
      .email("Invalid email address")
      .required("Email is required"),
  });

  const handleSubmit = async (values: { email: string }) => {
    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/forgot-password`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: values.email,
          }),
        }
      );

      if (response.ok) {
        toast.success(
          "If this email exists in our system, you'll receive a password reset link"
        );
        setEmailSent(true);
      } else {
        toast.error("Failed to send reset email. Please try again.");
      }
    } catch (error) {
      toast.error("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-center mb-8"
      >
        <div className="flex justify-center mb-4">
          <div className="bg-indigo-100 p-3 rounded-full">
            <div className="bg-indigo-600 text-white p-3 rounded-full">
              <FiMail size={28} />
            </div>
          </div>
        </div>

        {emailSent ? (
          <>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Check Your Email
            </h2>
            <p className="text-gray-600 mb-6">
              We've sent a password reset link to your email address
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-left">
              <p className="text-blue-800">
                <span className="font-medium">Didn't receive the email?</span>
                <br />
                Check your spam folder or resend the link
              </p>
            </div>
          </>
        ) : (
          <>
            <h2 className="text-3xl font-bold text-gray-900 mb-2 ">
              Forgot Password
            </h2>
            <p className="text-gray-600">
              Enter your email to reset your password
            </p>
          </>
        )}
      </motion.div>

      {!emailSent ? (
        <Formik
          initialValues={{ email: "" }}
          validationSchema={validationSchema}
          onSubmit={handleSubmit}
        >
          {({ values, errors, touched, handleChange }) => (
            <Form className="w-[40%] mx-auto">
              <AuthInput
                icon={<FiMail />}
                name="email"
                type="email"
                placeholder="Email address"
                value={values.email}
                onChange={handleChange}
                error={touched.email && errors.email}
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
                    <div className="flex items-center justify-center w-2.5">
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
                      Sending...
                    </div>
                  ) : (
                    "Send Reset Link"
                  )}
                </button>
              </motion.div>
            </Form>
          )}
        </Formik>
      ) : (
        ""
      )}

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="text-center shadow-2xl"
      >
        <Link
          href="/signin"
          className="inline-flex items-center text-indigo-600 hover:text-indigo-800 font-medium transition-colors"
        >
          <FiArrowLeft className="mr-2 " />
          Back to Sign In
        </Link>
      </motion.div>
    </>
  );
};

export default ForgotPassword;
