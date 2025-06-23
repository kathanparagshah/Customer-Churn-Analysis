# Deployment Guide

This guide provides step-by-step instructions for deploying the Customer Churn Analysis application to production.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (Vercel)      │───▶│   (Render)      │───▶│   (Optional)    │
│   React + Vite  │    │   FastAPI       │    │   PostgreSQL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Required Accounts
- [GitHub](https://github.com) - Source code repository
- [Vercel](https://vercel.com) - Frontend hosting
- [Render](https://render.com) - Backend hosting
- [Docker Hub](https://hub.docker.com) - Container registry (optional)

### Required Tools
- Git
- Node.js 18+
- Python 3.9+
- Docker (optional)
- Vercel CLI (optional)

## Backend Deployment

### Option 1: Deploy to Render (Recommended)

1. **Fork the Repository**
   ```bash
   # Fork https://github.com/kathanparagshah/Customer-Churn-Analysis
   git clone https://github.com/YOUR_USERNAME/Customer-Churn-Analysis.git
   cd Customer-Churn-Analysis
   ```

2. **Create Render Service**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name**: `churn-api`
     - **Environment**: `Docker`
     - **Region**: Choose closest to your users
     - **Branch**: `main`
     - **Dockerfile Path**: `./Dockerfile`
     - **Docker Context**: `.`

3. **Environment Variables**
   ```
   PORT=8000
   PYTHONPATH=/app
   LOG_LEVEL=INFO
   ```

4. **Advanced Settings**
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: Enable
   - **Plan**: Starter (free tier available)

5. **Custom Domain (Optional)**
   - Add custom domain: `api.customer-churn-demo.com`
   - Configure DNS records as instructed

### Option 2: Deploy to Heroku

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy to Heroku**
   ```bash
   heroku login
   heroku create your-churn-api
   heroku container:push web
   heroku container:release web
   heroku open
   ```

3. **Configure Environment Variables**
   ```bash
   heroku config:set PYTHONPATH=/app
   heroku config:set LOG_LEVEL=INFO
   ```

### Option 3: Docker Deployment

1. **Build and Run Locally**
   ```bash
   docker build -t churn-api .
   docker run -p 8000:8000 churn-api
   ```

2. **Push to Registry**
   ```bash
   docker tag churn-api ghcr.io/YOUR_USERNAME/churn-api:latest
   docker push ghcr.io/YOUR_USERNAME/churn-api:latest
   ```

## Frontend Deployment

### Option 1: Deploy to Vercel (Recommended)

1. **Automatic Deployment**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Configure project:
     - **Framework Preset**: Vite
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist`
     - **Install Command**: `npm install`

2. **Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add:
     ```
     VITE_API_BASE_URL=https://your-api-domain.com
     ```

3. **Custom Domain (Optional)**
   - Go to Project Settings → Domains
   - Add: `customer-churn-analysis-kgz3.vercel.app`

### Option 2: Manual Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy**
   ```bash
   cd frontend
   vercel login
   vercel --prod
   ```

### Option 3: Static Hosting

1. **Build the Project**
   ```bash
   cd frontend
   npm install
   VITE_API_BASE_URL=https://your-api-domain.com npm run build
   ```

2. **Deploy to Any Static Host**
   - Upload `dist/` folder to:
     - Netlify
     - GitHub Pages
     - AWS S3 + CloudFront
     - Firebase Hosting

## CI/CD Setup

### GitHub Secrets Configuration

1. **Backend Secrets**
   ```
   RENDER_SERVICE_ID=srv-xxxxxxxxxxxxx
   RENDER_API_KEY=rnd_xxxxxxxxxxxxx
   ```

2. **Frontend Secrets**
   ```
   VERCEL_TOKEN=xxxxxxxxxxxxx
   VERCEL_ORG_ID=team_xxxxxxxxxxxxx
   VERCEL_PROJECT_ID=prj_xxxxxxxxxxxxx
   ```

### Getting Render Credentials

1. **Service ID**
   - Go to your Render service dashboard
   - Copy the service ID from the URL: `srv-xxxxxxxxxxxxx`

2. **API Key**
   - Go to Account Settings → API Keys
   - Create new API key
   - Copy the key: `rnd_xxxxxxxxxxxxx`

### Getting Vercel Credentials

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   vercel login
   ```

2. **Link Project**
   ```bash
   cd frontend
   vercel link
   ```

3. **Get Project Info**
   ```bash
   cat .vercel/project.json
   ```

4. **Get Token**
   - Go to Vercel Account Settings → Tokens
   - Create new token
   - Copy the token

## Environment Variables

### Backend Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port | 8000 | No |
| `PYTHONPATH` | Python module path | /app | Yes |
| `LOG_LEVEL` | Logging level | INFO | No |
| `MODEL_PATH` | Path to model file | ../models/churn_model.pkl | No |

### Frontend Environment Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `VITE_API_BASE_URL` | Backend API URL | https://api.customer-churn-demo.com | Yes |

## Verification

### Backend Health Check

```bash
# Check if backend is running
curl https://your-api-domain.com/health

# Expected response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime": "0:05:23",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Frontend Functionality

1. **Visit your frontend URL**
2. **Upload sample data**:
   - Use `frontend/sample_data.csv`
   - Verify predictions are displayed
   - Check browser console for errors

### End-to-End Test

```bash
# Test the prediction endpoint
curl -X POST https://your-api-domain.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "CreditScore": 650,
      "Geography": "France",
      "Gender": "Female",
      "Age": 35,
      "Tenure": 5,
      "Balance": 50000,
      "NumOfProducts": 2,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 75000
    }]
  }'
```

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Verify frontend URL is in CORS allowed origins
   - Check browser console for specific error messages
   - Ensure API URL is correct in frontend environment variables

2. **Build Failures**
   - Check GitHub Actions logs
   - Verify all dependencies are in requirements.txt/package.json
   - Ensure environment variables are set correctly

3. **Model Loading Issues**
   - Verify model file exists in the repository
   - Check file paths in the application
   - Review application logs for specific errors

4. **API Connection Issues**
   - Verify backend is running and accessible
   - Check network connectivity
   - Verify SSL certificates for HTTPS endpoints

### Debugging Commands

```bash
# Check backend logs (Render)
render logs --service-id srv-xxxxxxxxxxxxx

# Check frontend build logs (Vercel)
vercel logs https://your-frontend-domain.com

# Test API locally
docker run -p 8000:8000 churn-api
curl http://localhost:8000/health

# Test frontend locally
cd frontend
npm run dev
```

## Security Considerations

1. **API Security**
   - CORS is configured for specific origins only
   - Consider adding rate limiting
   - Implement API authentication if needed

2. **Environment Variables**
   - Never commit secrets to version control
   - Use platform-specific secret management
   - Rotate API keys regularly

3. **HTTPS**
   - Both Render and Vercel provide HTTPS by default
   - Ensure all API calls use HTTPS in production

## Monitoring

1. **Backend Monitoring**
   - Render provides built-in metrics
   - Health check endpoint: `/health`
   - Metrics endpoint: `/metrics` (Prometheus format)

2. **Frontend Monitoring**
   - Vercel provides analytics and performance metrics
   - Monitor Core Web Vitals
   - Set up error tracking (Sentry, LogRocket)

3. **Alerts**
   - Configure uptime monitoring (UptimeRobot, Pingdom)
   - Set up error rate alerts
   - Monitor API response times

## Cost Optimization

1. **Free Tiers**
   - Render: 750 hours/month free
   - Vercel: Unlimited for personal projects
   - GitHub Actions: 2000 minutes/month free

2. **Scaling**
   - Start with free tiers
   - Monitor usage and upgrade as needed
   - Consider serverless options for variable workloads

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review platform-specific documentation
3. Open an issue in the GitHub repository
4. Contact platform support if needed