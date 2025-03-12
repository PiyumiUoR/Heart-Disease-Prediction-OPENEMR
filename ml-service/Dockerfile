# Use an official lightweight Python image
FROM python:3.9

# Install netcat for waiting script
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the ML service files
COPY model.py .
COPY requirements.txt .
COPY train_values.csv .
COPY Train_Labels.csv .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and give execute permission to the wait script
COPY wait-for-kafka.sh /wait-for-kafka.sh
RUN chmod +x /wait-for-kafka.sh

# Expose the port (if needed for an API service)
EXPOSE 5001

# Run the script when the container starts
CMD ["/wait-for-kafka.sh", "python", "model.py"]
