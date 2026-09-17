FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# One command starts both the API (background) and the Streamlit UI (foreground).
CMD ["python", "run.py"]
