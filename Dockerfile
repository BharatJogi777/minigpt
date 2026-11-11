FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY minigpt ./minigpt
COPY artifacts ./artifacts
COPY checkpoints ./checkpoints
EXPOSE 8000
CMD ["uvicorn","minigpt.api:app","--host","0.0.0.0","--port","8000"]
