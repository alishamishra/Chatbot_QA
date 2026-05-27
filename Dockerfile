FROM python:3.12-slim

RUN apt-get update && apt-get install -y libxcb1 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1



# 1. Set the working directory
WORKDIR /app

# 2. Copy files (they will go into /app because of the line above)
COPY requirements.txt .
COPY chatbot_core.py .
COPY app.py .

RUN pip install -r requirements.txt

RUN pip install opencv-python-headless

# 3. Inform Docker of the port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
