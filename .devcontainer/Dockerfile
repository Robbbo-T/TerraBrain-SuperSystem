# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose necessary ports
EXPOSE 8000
