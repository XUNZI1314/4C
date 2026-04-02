#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=protein-visualizer
CONTAINER_NAME=protein-visualizer

echo "Building Docker image: ${IMAGE_NAME} ..."
docker build -t "${IMAGE_NAME}" .

EXISTING=$(docker ps -q -f name="${CONTAINER_NAME}")
if [ -n "${EXISTING}" ]; then
  echo "Stopping existing container ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" || true
  docker rm "${CONTAINER_NAME}" || true
fi

echo "Starting container ${CONTAINER_NAME}..."
docker run -d --name "${CONTAINER_NAME}" -p 8501:8501 -v "$(pwd)/data:/app/data" "${IMAGE_NAME}"

echo "应用已启动: http://localhost:8501"
