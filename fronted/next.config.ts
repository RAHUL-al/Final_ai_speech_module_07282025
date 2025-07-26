import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://llm.edusmartai.com/api:path*",
      },
    ];
  },
};

export default nextConfig;
