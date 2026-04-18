# 本地 Docker 安装与运行指南

本文档说明如何在本地构建并运行 ProteinInsight Docker 服务。

## 前提

- Windows 10/11、Linux 或 macOS。
- 已安装 Docker Desktop 或 Docker Engine。
- Windows 推荐启用 WSL2 后端。
- 仓库根目录包含 `Dockerfile` 和 `docker-compose.yml`。

## 验证 Docker

```powershell
docker --version
docker run --rm hello-world
```

如果 `hello-world` 能正常运行，说明 Docker 基础环境可用。

## 使用脚本运行

仓库根目录提供：

- `run_docker.ps1`：Windows PowerShell。
- `run_docker.sh`：Linux / macOS / Git Bash / WSL。

Windows：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\run_docker.ps1
```

Linux / macOS / WSL：

```bash
chmod +x run_docker.sh
./run_docker.sh
```

脚本行为：

- 构建镜像：`docker build -t protein-visualizer .`
- 停止并删除同名旧容器。
- 启动新容器。
- 映射端口 `8501:8501`。
- 挂载本地 `./data` 到容器 `/app/data`。

## 手动构建与运行

PowerShell：

```powershell
docker build -t protein-visualizer .
docker run -d --name protein-visualizer -p 8501:8501 -v ${PWD}\data:/app/data protein-visualizer
```

Linux / macOS：

```bash
docker build -t protein-visualizer .
docker run -d --name protein-visualizer -p 8501:8501 -v "$PWD/data:/app/data" protein-visualizer
```

访问：

```text
http://localhost:8501
```

## 使用 Docker Compose

```powershell
docker compose up --build -d
```

查看日志：

```powershell
docker compose logs -f
```

停止服务：

```powershell
docker compose down
```

## 从 GHCR 拉取镜像

如果 CI 已发布镜像到 GitHub Container Registry：

```bash
docker pull ghcr.io/<OWNER>/protein-visualizer:latest
docker run -d --name protein-visualizer -p 8501:8501 ghcr.io/<OWNER>/protein-visualizer:latest
```

私有镜像需要先登录：

```bash
echo $GHCR_PAT | docker login ghcr.io -u <USERNAME> --password-stdin
```

## 常见问题

- `docker` 命令找不到：确认 Docker Desktop 已安装并启动。
- 端口冲突：停止占用 `8501` 的程序，或修改 compose / run 脚本中的端口映射。
- Windows 挂载失败：确认 Docker Desktop 有权限访问当前磁盘。
- 页面历史无法保存：确认本地 `data/` 目录存在且可写。
- P2Rank 不可用：容器内默认不包含 P2Rank 二进制；需要自行安装或挂载，并设置 `P2RANK_HOME` / `P2RANK_SCRIPT`。

## 停止并删除容器

```bash
docker stop protein-visualizer || true
docker rm protein-visualizer || true
```
