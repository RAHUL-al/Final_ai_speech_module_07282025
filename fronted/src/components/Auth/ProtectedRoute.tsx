"use client";

import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";
import useAuth from "@/hooks/useAuth";

const ProtectedRoute = ({ children }) => {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, loading } = useAuth();

  useEffect(() => {
    const isPasswordResetRoute = pathname.startsWith("/forgotten-password");

    if (isPasswordResetRoute) return;

    if (!loading && !isAuthenticated) {
      router.replace("/signin");
    }
  }, [isAuthenticated, loading, router, pathname]);

  if (
    !pathname.startsWith("/forgotten-password") &&
    (loading || !isAuthenticated)
  ) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-600"></div>
          <p className="mt-4 text-lg font-medium text-gray-700">Loading..</p>
        </div>
      </div>
    );
  }

  return isAuthenticated ? children : null;
};

export default ProtectedRoute;
