# Use the Python image as the base image
FROM python:3.10

# Working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run rl_main.py when the container launches
CMD ["python", "rl_main.py"]
