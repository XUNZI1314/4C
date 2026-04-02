Param(
  [string]$ImageName = "protein-visualizer",
  [string]$ContainerName = "protein-visualizer"
)

Write-Host "Building Docker image: $ImageName ..."
docker build -t $ImageName .

$existing = docker ps -q -f name=$ContainerName
if ($existing) {
  Write-Host "Stopping existing container $ContainerName..."
  docker stop $ContainerName | Out-Null
  docker rm $ContainerName | Out-Null
}

Write-Host "Starting container $ContainerName..."
docker run -d --name $ContainerName -p 8501:8501 -v ${PWD}\data:/app/data $ImageName

Write-Host "应用已启动: http://localhost:8501"
