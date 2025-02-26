# Use an official lightweight Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy the ML service files
COPY model.py .
COPY requirements.txt .
COPY train_values.csv .
COPY Train_Labels.csv .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (if needed for an API service)
EXPOSE 5001

# Run the script when the container starts
CMD ["python", "model.py"]
