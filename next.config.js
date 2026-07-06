/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/:path*', // Proxy to FastAPI backend
      },
      {
        source: '/reports/:path*',
        destination: 'http://127.0.0.1:8000/reports/:path*', // Proxy report downloads
      },
    ]
  },
}

module.exports = nextConfig
