"use client";
import React, { useState } from "react";
import { FiUser, FiMail, FiLock } from "react-icons/fi";
import { Formik, Form } from "formik";
import * as Yup from "yup";
import { toast } from "react-hot-toast";
import AuthLayout from "./AuthLayout";
import AuthInput from "@/components/Auth/AuthInput";
import PasswordStrength from "@/components/Auth/PasswordStrength";
import { registerUser } from "@/store/slices/authSlice";
import { useAppDispatch } from "@/store";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { FormikHelpers } from "formik";
import { motion } from "framer-motion";

interface FormValues {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

const SignUp = () => {
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");
  const dispatch = useAppDispatch();
  const router = useRouter();
  // const { isLoading, error } = useAppSelector((state: any) => state.auth);

  const validationSchema = Yup.object({
    // name: Yup.string().required("Name is required"),
    email: Yup.string().email("Invalid email").required("Email is required"),
    username: Yup.string().required("Username is required"),
    // phone: Yup.string().matches(/^[0-9]{10}$/, "Invalid phone number"),
    password: Yup.string()
      .min(8, "Password must be at least 8 characters")
      .required("Password is required"),
    confirmPassword: Yup.string()
      .oneOf([Yup.ref("password"), null], "Passwords must match")
      .required("Confirm password is required"),
  });

  const handleSubmit = async (
    values: FormValues,
    { setSubmitting }: FormikHelpers<FormValues>
  ) => {
    setLoading(true);
    setApiError("");

    const params = {
      email: values.email,
      password: values.password,
      confirm_password: values.confirmPassword,
      username: values.username,
    };

    try {
      const result = await dispatch(registerUser(params));
      if (registerUser.fulfilled.match(result)) {
        toast.success("Account created successfully!");
        router.push("/signin");
      } else if (registerUser.rejected.match(result)) {
        // Extract error message from payload
        console.log("123456782345678923456789", result);
        const errorMessage =
          result.payload?.message || "Error creating account";
        setApiError(errorMessage);
        toast.error(errorMessage);
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "An unexpected error occurred";
      setApiError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
      setSubmitting(false);
    }
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
          <motion.div
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{
              type: "spring",
              stiffness: 200,
              damping: 15,
            }}
            className="bg-indigo-100 p-3 rounded-full"
          >
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-3 rounded-full">
              <FiUser size={28} />
            </div>
          </motion.div>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Create your account
        </h2>
        <p className="text-gray-600">
          Join thousands of users managing their accounts securely
        </p>
      </motion.div>

      <Formik
        initialValues={{
          username: "",
          email: "",
          password: "",
          confirmPassword: "",
        }}
        validationSchema={validationSchema}
        onSubmit={handleSubmit}
      >
        {({ values, errors, touched, handleChange }) => (
          <Form>
            {apiError && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg"
              >
                {apiError}
              </motion.div>
            )}

            <AuthInput
              icon={<FiUser />}
              name="username"
              type="text"
              placeholder="Username"
              value={values.username}
              onChange={handleChange}
              error={touched.username && errors.username}
            />

            <AuthInput
              icon={<FiMail />}
              name="email"
              type="email"
              placeholder="Email Address"
              value={values.email}
              onChange={handleChange}
              error={touched.email && errors.email}
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

            <PasswordStrength password={values.password} />

            <AuthInput
              icon={<FiLock />}
              name="confirmPassword"
              type="password"
              placeholder="Confirm Password"
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
                    Creating Account...
                  </div>
                ) : (
                  "Create Account"
                )}
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="text-center text-gray-600 text-sm"
            >
              <p className="mb-2">
                By signing up, you agree to our{" "}
                <Link
                  href="#"
                  className="font-medium text-indigo-600 hover:text-indigo-800"
                >
                  Terms
                </Link>{" "}
                and{" "}
                <Link
                  href="#"
                  className="font-medium text-indigo-600 hover:text-indigo-800"
                >
                  Privacy Policy
                </Link>
              </p>
              <p>
                Already have an account?{" "}
                <Link
                  href="/signin"
                  className="font-medium text-indigo-600 hover:text-indigo-800"
                >
                  Sign In
                </Link>
              </p>
            </motion.div>
          </Form>
        )}
      </Formik>
    </AuthLayout>
  );
};

export default SignUp;
