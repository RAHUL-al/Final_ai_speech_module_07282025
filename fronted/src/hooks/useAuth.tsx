// src/hooks/useAuth.js
"use client";

import { useState, useEffect } from "react";
import { AppDispatch, useAppSelector } from "@/store";
import { useDispatch } from "react-redux";
import { verifyAuth } from "@/store/slices/authSlice";

const useAuth = () => {
  const [loading, setLoading] = useState(true);
  const isAuthenticated = useAppSelector((state) => state.auth.isAuthenticated);

  const dispatch = useDispatch<AppDispatch>();
  console.log(isAuthenticated);
  useEffect(() => {
    const checkAuth = async () => {
      try {
        await dispatch(verifyAuth()).unwrap();
      } catch (error) {
        console.error("Auth verification failed:", error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, [dispatch]);

  return { loading, isAuthenticated };
};

export default useAuth;
