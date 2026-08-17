FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        langgraph>=1.2.10 \
        langchain-openai \
        langchain-core \
        langchain-community \
        fastapi>=0.115.0 \
        uvicorn>=0.30.0 \
        redis>=8.1.0 \
        ddgs>=9.14.4 \
        python-dotenv \
        pydantic>=2.0

COPY . .

EXPOSE 8000

CMD ["python", "-u", "main.py"]
