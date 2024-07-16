# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME PP-TEBDE

# Run app.py when the container launches
CMD ["python", "main.py"]

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Add labels
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Privacy-Preserving Temporal Exposure Bias Detection and Explanation (PP-TEBDE)"

# Add volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Set up a non-root user
RUN useradd -m myuser
USER myuser

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Set timezone
ENV TZ=UTC

# Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Remove unnecessary files
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
