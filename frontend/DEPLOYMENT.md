# Frontend Deployment Guide

## Vercel Deployment

This React frontend is configured for easy deployment on Vercel.

### Quick Deploy

1. **Visit Vercel**: Go to [vercel.com](https://vercel.com)
2. **Sign up/Login**: Use your GitHub account
3. **Import Project**: Click "New Project" and import this repository
4. **Configure Settings**:
   - Framework Preset: Vite
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install`

### Manual Deployment

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy from frontend directory**:
   ```bash
   cd frontend
   vercel --prod
   ```

### Environment Variables

If you need to configure the backend API URL for production:

1. In Vercel dashboard, go to your project settings
2. Add environment variable:
   - Name: `VITE_API_URL`
   - Value: Your deployed backend URL (e.g., `https://your-backend.herokuapp.com`)

### Configuration Files

- `vercel.json`: Vercel-specific configuration
- `vite.config.js`: Vite build configuration
- `package.json`: Build scripts and dependencies

### Features

- ✅ Single Page Application (SPA) routing
- ✅ Optimized production build
- ✅ Automatic deployments from GitHub
- ✅ Custom domain support
- ✅ HTTPS by default

### Troubleshooting

- Ensure all dependencies are in `package.json`
- Check build logs in Vercel dashboard
- Verify API endpoints are accessible from production