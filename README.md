# HackRx API Deployment Guide

## Quick Setup

1. **Create your environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## Deployment Options

### 1. Render (Recommended - Free Tier)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) and create account
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `GEMINI_API_KEY`: Your Google AI Studio API key
   - `HACKRX_API_KEY`: Your HackRx competition token
7. Deploy!

Your endpoint will be: `https://your-app-name.onrender.com/hackrx/run`

### 2. Railway

1. Go to [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python and installs dependencies
5. Add environment variables in Variables tab
6. Deploy!

### 3. Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
3. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set GEMINI_API_KEY=your_key
   heroku config:set HACKRX_API_KEY=your_token
   git push heroku main
   ```

### 4. DigitalOcean App Platform

1. Go to DigitalOcean App Platform
2. Create new app from GitHub
3. Configure Python app
4. Add environment variables
5. Deploy

### 5. Google Cloud Run

1. Build Docker image:
   ```bash
   docker build -t hackrx-api .
   docker tag hackrx-api gcr.io/PROJECT_ID/hackrx-api
   docker push gcr.io/PROJECT_ID/hackrx-api
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy hackrx-api \
     --image gcr.io/PROJECT_ID/hackrx-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GEMINI_API_KEY=your_key,HACKRX_API_KEY=your_token
   ```

## Testing Your Deployment

Test your endpoint with curl:

```bash
curl -X POST "https://your-app.onrender.com/hackrx/run" \
  -H "Authorization: Bearer your_hackrx_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payments?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

## Environment Variables Required

- `GEMINI_API_KEY`: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- `HACKRX_API_KEY`: Provided by HackRx competition organizers

## Performance Tips

- Use platforms with at least 1GB RAM (Render, Railway free tiers work)
- The app uses lazy loading to minimize memory usage
- First request might be slower due to model loading
- Subsequent requests will be faster

## Troubleshooting

- **Memory errors**: Try Render or Railway (better free tier specs)
- **Timeout errors**: Increase timeout settings in your deployment platform
- **API errors**: Check your environment variables are set correctly
- **PDF processing errors**: Ensure the document URL is accessible

## Ready for Submission

Once deployed, your endpoint will be:
`https://your-app-name.platform.com/hackrx/run`

Submit this URL to the HackRx competition platform!