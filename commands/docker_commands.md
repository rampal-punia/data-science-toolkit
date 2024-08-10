# Comprehensive Docker Tutorial for Data Scientists

## Introduction to Docker
Docker is a platform for developing, shipping, and running applications in containers. For data scientists, Docker provides a way to create reproducible environments, package applications with their dependencies, and ensure consistency across different systems.

## Basic Docker Commands

### Check Docker Installation

```bash
# Check Docker version
docker --version

# View detailed Docker information
docker info
```

### Docker Daemon Management

```bash
# Check Docker daemon status
sudo systemctl status docker
# or
sudo service docker status

# Start Docker daemon
sudo systemctl start docker
# or
sudo service docker start

# Stop Docker daemon
sudo systemctl stop docker
# or
sudo service docker stop

# Restart Docker daemon
sudo systemctl restart docker
# or
sudo service docker restart
```

## Working with Docker Images

### Pulling Images

```bash
# Pull an image from Docker Hub
docker pull <image_name>:<tag>

# Example: Pull the latest Python image
docker pull python:latest

# Example: Pull a specific version of TensorFlow
docker pull tensorflow/tensorflow:2.7.0-gpu
```

### Listing Images

```bash
# List all Docker images on your system
docker images

# List images with specific format
docker images --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

### Removing Images

```bash
# Remove a specific image
docker rmi <image_id>

# Remove all images
docker rmi $(docker images -q)

# Force remove all images
docker rmi -f $(docker images -q)

# Remove dangling images (images with no tags or references)
docker image prune -f
```

## Working with Docker Containers

### Running Containers

```bash
# Run a container from an image
docker run <image_name>

# Run a container in detached mode (background)
docker run -d <image_name>

# Run a container with a specific name
docker run --name my_container <image_name>

# Run a container and map a port
docker run -p host_port:container_port <image_name>

# Example: Run a Jupyter Notebook server
docker run -p 8888:8888 jupyter/datascience-notebook
```

### Managing Containers

```bash
# List running containers
docker ps

# List all containers (including stopped ones)
docker ps -a

# Stop a running container
docker stop <container_id>

# Start a stopped container
docker start <container_id>

# Remove a container
docker rm <container_id>

# Remove all stopped containers
docker container prune
```

### Executing Commands in Containers

```bash
# Execute a command in a running container
docker exec <container_id> <command>

# Open an interactive shell in a running container
docker exec -it <container_id> /bin/bash
```

## Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications.

### Basic Docker Compose Commands

```bash
# Start services defined in docker-compose.yml
docker-compose up

# Start services in detached mode
docker-compose up -d

# Build or rebuild services
docker-compose build

# Stop and remove containers, networks, images, and volumes
docker-compose down

# View logs of services
docker-compose logs
```

### Example docker-compose.yml for Data Science

```yaml
version: '3'
services:
  jupyter:
    image: jupyter/datascience-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: mysecretpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

To use this, save it as `docker-compose.yml` and run:

```bash
docker-compose up -d
```

This will start a Jupyter notebook server and a PostgreSQL database, which is a common setup for data science projects.

## Best Practices for Data Scientists

1. Use specific tags for images to ensure reproducibility.
2. Use volumes to persist data and share it between the host and containers.
3. Use Docker Compose for complex setups involving multiple services.
4. Create custom Dockerfiles for your projects to encapsulate all dependencies.
5. Use .dockerignore files to exclude unnecessary files from your Docker context.

## More Topics

### Building Custom Images

Create a Dockerfile:

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "your_script.py"]
```

Build the image:

```bash
docker build -t my_data_science_app .
```

### Using GPU with Docker

To use GPU in Docker containers, install NVIDIA Container Toolkit and use the `--gpus` flag:

```bash
# Run a TensorFlow container with GPU support
docker run --gpus all -it tensorflow/tensorflow:latest-gpu python
```

Remember to adjust commands as needed for your specific use case and system configuration. Docker is a powerful tool that can greatly enhance reproducibility and deployment in data science projects.