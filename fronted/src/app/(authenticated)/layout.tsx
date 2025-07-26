import ProtectedRoute from "@/components/Auth/ProtectedRoute";
import Layout from "@/components/Layout/Layout";
import React from "react";
export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ProtectedRoute>
      <Layout>{children}</Layout>
    </ProtectedRoute>
  );
}
