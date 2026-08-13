# AI Travel Planner

## FastAPI app

The project now exposes a FastAPI service for travel planning.

### Run locally

```bash
cd /home/brainerhub/Downloads/AI_travel_planner
/home/brainerhub/Downloads/AI_travel_planner/.venv/bin/python main.py
```

Then open:

- http://localhost:8000/ for health check
- http://localhost:8000/docs for Swagger UI
- POST http://localhost:8000/plan with JSON body for trip planning

Example request:

```json
{
  "destination": "Paris",
  "origin": "London",
  "budget": "1500",
  "currency": "EUR"
}
```

## Docker Compose

This project includes a `docker-compose.yml` that starts:

- `ai_travel_planner` (your app)
- `redis`

The app is exposed on host port `9001` and forwards to container port `8000`.

### Run with Docker

```bash
docker compose up --build -d
```

### Access

Open: http://localhost:9001

Portainer note: Portainer may show containers with stack-based names (for example `stackname_service_1`). Avoid using a fixed `container_name` in the compose file so Portainer can manage containers and updates cleanly.

### Notes

- Port `9000` is commonly used by Portainer, so this compose file uses `9001` instead.
- Redis is included as a service and available at `redis:6379` inside the compose network.
