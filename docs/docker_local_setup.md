# 本地 Docker 安装与运行指南

本文档说明如何在本地（Windows）安装 Docker 并使用仓库里的一键脚本构建与启动 `protein-visualizer` 服务。

## 前提
- Windows 10/11（推荐）或等效的 Windows Server
- 管理员权限以便安装软件
- 如果使用 WSL2（推荐），请确保已启用 WSL2 和安装了一个 Linux 发行版

## 安装 Docker Desktop（Windows）
1. 访问 Docker 官方下载页面并下载安装包（搜索 “Docker Desktop”）。
2. 运行安装包并按照提示完成安装。安装时选择启用 WSL 2 后端（如果可用）。
3. 安装完成后重启计算机（如果安装程序提示）。
4. 启动 Docker Desktop，等待 Docker 引擎启动（系统托盘图标变为绿/正常状态）。

验证安装：在 PowerShell 中运行：

```powershell
docker --version
docker run --rm hello-world
```

如果 `hello-world` 成功运行，说明 Docker 已能正常拉取并运行容器。

## 使用仓库一键脚本（推荐）
仓库根目录下已提供两个脚本：`run_docker.ps1`（Windows / PowerShell）和 `run_docker.sh`（Linux / macOS / Git Bash / WSL）。

在 PowerShell（以仓库根目录为当前目录）运行：

```powershell
# 允许当前会话运行本地脚本（如果需要）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# 运行脚本并构建+启动容器
.\run_docker.ps1
```

在 Git Bash / WSL / Linux 中运行：

```bash
# 可执行权限（如果需要）
chmod +x run_docker.sh
./run_docker.sh
```

脚本行为（摘要）：
- 构建镜像：`docker build -t protein-visualizer .`
- 停止并删除同名已运行容器（如存在）
- 启动新容器，映射端口 `8501:8501` 并挂载本地 `./data` 到容器 `/app/data`

你也可手动执行相同命令：

```powershell
# 手动构建
docker build -t protein-visualizer .
# 手动运行（PowerShell）
docker run -d --name protein-visualizer -p 8501:8501 -v ${PWD}\\data:/app/data protein-visualizer
```

或使用 `docker compose`（仓库含 `docker-compose.yml`）：

```bash
# 使用 Docker Compose (较新 Docker 使用 `docker compose`)
docker compose up --build -d
# 或旧命令
# docker-compose up --build -d
```

访问： http://localhost:8501

## 从 CI（GHCR）拉取已发布镜像并运行
如你已将代码推送到 GitHub 且 CI workflow（`.github/workflows/ci-docker-publish.yml`）成功运行，镜像会被推送到 GitHub Container Registry（GHCR）。拉取并运行：

```bash
# 把 <OWNER> 替换为你的 GitHub 组织或用户名称
docker pull ghcr.io/<OWNER>/protein-visualizer:latest
docker run -d --name protein-visualizer -p 8501:8501 ghcr.io/<OWNER>/protein-visualizer:latest
```

如果仓库或镜像为私有，需要先登录 GHCR：

```bash
# 在终端中使用个人访问令牌 (PAT) 登录
echo $GHCR_PAT | docker login ghcr.io -u <USERNAME> --password-stdin
```

## 常见问题与排查
- 如果 `docker` 命令未找到：确认 Docker Desktop 已安装并已启动。
- 如果端口被占用：停止占用端口的程序或修改 `run_docker`/`docker-compose.yml` 中的端口映射。
- 权限问题（Windows）：以管理员权限运行 PowerShell 或启用开发人员模式。

## 完成后
1. 打开浏览器访问 `http://localhost:8501`。
2. 若需停止并移除容器：

```bash
docker stop protein-visualizer || true
docker rm protein-visualizer || true
```

---
如果你希望我把该文档内容合并到 `README.md` 的 Docker 部分，或生成一个 GitHub Actions 的 badge 与发布说明，我可以继续处理。
