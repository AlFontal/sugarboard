# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The app will be available at http://localhost:8080

### Using Docker directly

```bash
# Build the image
docker build -t sugarboard:latest .

# Run the container
docker run -d \
  --name sugarboard \
  -p 8080:8080 \
  -e STORAGE_SECRET="$(python -c 'from secrets import token_hex; print(token_hex(32))')" \
  --restart unless-stopped \
  sugarboard:latest

# View logs
docker logs -f sugarboard

# Stop and remove
docker stop sugarboard
docker rm sugarboard
```

## Deployment Options

### 1. Deploy to Heroku with Docker

```bash
# Login to Heroku
heroku login
heroku container:login

# Create app (if needed)
heroku create your-app-name

# Build and push
heroku container:push web -a your-app-name
heroku container:release web -a your-app-name

# Open app
heroku open -a your-app-name
```

### 2. Deploy to DigitalOcean App Platform

1. Push your code to GitHub
2. Go to DigitalOcean App Platform
3. Create new app from GitHub repo
4. Select Dockerfile deployment
5. Set port to 8080
6. Deploy!

### 3. Deploy to Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/sugarboard

# Deploy
gcloud run deploy sugarboard \
  --image gcr.io/YOUR-PROJECT-ID/sugarboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

### 4. Deploy to AWS ECS/Fargate

```bash
# Build and tag
docker build -t sugarboard:latest .

# Tag for ECR
docker tag sugarboard:latest YOUR-ECR-REPO-URL:latest

# Push to ECR
docker push YOUR-ECR-REPO-URL:latest

# Create ECS task definition and service (via AWS Console or CLI)
```

### 5. Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch app
fly launch

# Deploy
fly deploy
```

## Configuration

### Environment Variables

You can pass environment variables in docker-compose.yml or via -e flag:

```yaml
environment:
  - CGM_SITE=https://your-site.herokuapp.com  # optional: pre-fills the UI form
  - STORAGE_SECRET=change-me-super-secret    # required for NiceGUI user storage
  - RECENT_REQUEST_TIMEOUT=60               # optional: bump Nightscout timeout
  - DISPLAY_TIMEZONE=UTC
  - TZ=UTC
  - ALLOW_HTTP=0
  - ALLOW_INSECURE_NS_URLS=0
```

> The Nightscout read token or API secret is entered at runtime through the dashboard's "Nightscout Connection" card; do not bake it into the container environment.

Keep `.env`, `.cache/`, `.nicegui/`, and exported CGM data out of git and Docker build contexts. These files can contain credentials or protected health data.

### Public access

Sugarboard is designed as a personal dashboard. If you expose it beyond a trusted private network, put it behind reverse-proxy authentication such as Nginx basic auth, Traefik ForwardAuth, Authelia, or an equivalent gateway. NiceGUI's password option is acceptable for a quick personal gate, but a proxy is easier to audit at the edge.

### Nightscout API secret

Nightscout's legacy API-secret protocol requires SHA1. Sugarboard keeps that behavior for compatibility, but read-only Nightscout tokens are preferred.

### Persistent Cache & Credentials

To persist the cache and NiceGUI user storage (Nightscout credentials), mount volumes:

```bash
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/cache_data:/app/.cache \
  -v $(pwd)/storage_data:/app/.nicegui \
  sugarboard:latest
```

## Health Check

The container includes a health check at `/health` endpoint. You can verify it's running:

```bash
curl http://localhost:8080/health
# Should return: ok
```

## Troubleshooting

### Check logs
```bash
docker logs -f sugarboard
```

### Interactive shell
```bash
docker exec -it sugarboard /bin/bash
```

### Rebuild without cache
```bash
docker-compose build --no-cache
```
