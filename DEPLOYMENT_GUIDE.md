# Deployment Guide: Connecting Frontend to Real APIs

This guide will help you deploy the Customer Churn Analysis application with real API integration instead of mock data.

## Overview

Currently, the frontend uses mock data for predictions. This guide shows how to:
1. Deploy the FastAPI backend
2. Configure environment variables in Vercel
3. Connect the frontend to real APIs
4. Test the integrated system

## Step 1: Deploy the FastAPI Backend

### Option A: Deploy to Render (Recommended)

1. **Create a Render account** at [render.com](https://render.com)

2. **Connect your GitHub repository** to Render

3. **Create a new Web Service** with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd deployment && python -m uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
   - **Instance Type**: Free tier is sufficient for testing

4. **Set Environment Variables** in Render:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

5. **Deploy** and note your backend URL (e.g., `https://your-app.onrender.com`)

### Option B: Deploy to Railway

1. **Create a Railway account** at [railway.app](https://railway.app)
2. **Connect your GitHub repository**
3. **Configure the start command**: `cd deployment && python -m uvicorn app:app --host 0.0.0.0 --port $PORT`
4. **Set environment variables** as needed
5. **Deploy** and note your backend URL

### Option C: Deploy to Heroku

1. **Create a Heroku account** and install Heroku CLI
2. **Create a Procfile** in the root directory:
   ```
   web: cd deployment && python -m uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
3. **Deploy using Git**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Step 2: Configure Vercel Environment Variables

1. **Go to your Vercel dashboard** at [vercel.com](https://vercel.com)

2. **Select your project** (customer-churn-analysis)

3. **Go to Settings > Environment Variables**

4. **Add the following environment variable**:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: Your deployed backend URL (e.g., `https://your-app.onrender.com`)
   - **Environment**: Production (and Preview if you want)

5. **Save the changes**

6. **Redeploy your frontend** to apply the new environment variable:
   - Go to Deployments tab
   - Click the three dots on the latest deployment
   - Select "Redeploy"

## Step 3: Verify API Integration

### Local Testing (Optional)

1. **Start the backend locally**:
   ```bash
   cd deployment
   python -m uvicorn app:app --reload --port 8000
   ```

2. **Start the frontend locally**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Test the integration** at `http://localhost:3000`

### Production Testing

1. **Visit your deployed frontend** at `https://customer-churn-analysis-kgz3.vercel.app`

2. **Test Single Prediction**:
   - Go to "Single Prediction" page
   - Fill in customer data
   - Click "Predict Churn"
   - Verify you get real predictions (not mock data)

3. **Test Batch Prediction**:
   - Go to "Batch Predictions" page
   - Upload a CSV file with customer data
   - Verify you get real batch predictions

4. **Test CSV Uploader**:
   - Use the CSV upload component
   - Verify predictions are processed correctly

## Step 4: API Endpoints Available

Your deployed backend provides these endpoints:

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch customer predictions
- `GET /model/info` - Model information
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Prometheus metrics

## Step 5: Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Ensure your Vercel domain is added to CORS origins in `deployment/app.py`
   - Current CORS settings include: `https://customer-churn-analysis-kgz3.vercel.app`

2. **Environment Variable Not Working**:
   - Verify `VITE_API_BASE_URL` is set in Vercel
   - Redeploy the frontend after adding environment variables
   - Check browser developer tools for the actual API calls being made

3. **API Errors**:
   - Check backend logs in your deployment platform
   - Verify the backend is running and accessible
   - Test API endpoints directly using the `/docs` page

4. **Model Loading Issues**:
   - Ensure model files are included in the deployment
   - Check that required dependencies are installed
   - Verify file paths in the backend code

### Debugging Steps

1. **Check API calls in browser**:
   - Open Developer Tools > Network tab
   - Look for API requests to your backend URL
   - Check for any 404, 500, or CORS errors

2. **Test backend directly**:
   - Visit `https://your-backend-url.com/docs`
   - Try making API calls directly from the documentation

3. **Check environment variables**:
   - In your frontend, add `console.log(import.meta.env.VITE_API_BASE_URL)` temporarily
   - Verify it shows your backend URL

## Step 6: Optional Enhancements

### Add Google OAuth Backend Endpoint

If you want to implement the Google OAuth backend endpoint:

1. **Add to `deployment/app.py`**:
   ```python
   @app.post("/auth/google")
   async def google_auth(user_data: dict):
       # Implement user data saving logic
       return {"success": True, "message": "User saved successfully"}
   ```

2. **Update CORS if needed** to include authentication headers

### Add Model Monitoring

1. **Set up logging** for prediction requests
2. **Add metrics collection** for model performance
3. **Implement alerts** for model drift or errors

## Success Criteria

Your deployment is successful when:

✅ Backend is deployed and accessible
✅ Frontend environment variables are configured
✅ Single predictions work with real API
✅ Batch predictions work with real API
✅ CSV upload works with real API
✅ No CORS errors in browser console
✅ API documentation is accessible at `/docs`

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review backend logs in your deployment platform
3. Test API endpoints directly using the `/docs` interface
4. Verify environment variables are set correctly

The application should now be fully integrated with real APIs instead of mock data!