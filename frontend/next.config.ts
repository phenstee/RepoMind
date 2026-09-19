import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Produces a self-contained .next/standalone server bundle so the Docker
  // runtime image does not need node_modules or the full source tree.
  // `next dev` / `npm start` outside Docker are unaffected.
  output: "standalone",
};

export default nextConfig;
