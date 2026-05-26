# Docker 容器化最佳实践

## 1. 镜像构建优化

### 1.1 选择正确的基础镜像

优先使用官方镜像的 slim 或 alpine 变体：

```dockerfile
# 推荐：slim 变体，镜像体积约 60MB
FROM python:3.12-slim

# 不推荐：完整镜像，体积约 1GB
FROM python:3.12
```

对于 Go 项目，推荐多阶段构建 + scratch 基础镜像：

```dockerfile
# 阶段一：编译
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app/server .

# 阶段二：运行
FROM alpine:3.19
RUN apk --no-cache add ca-certificates tzdata
COPY --from=builder /app/server /app/server
USER 1000:1000
ENTRYPOINT ["/app/server"]
```

### 1.2 层缓存优化

Docker 按行缓存，将有缓存的层放在前面，频繁变化的放在后面：

```dockerfile
# 优化顺序：依赖 → 源码
COPY requirements.txt .
RUN pip install -r requirements.txt    # 这层可以缓存
COPY . .                                # 源码变化时才重建
```

### 1.3 多阶段构建（Multi-stage Build）

```dockerfile
FROM node:20 AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

### 1.4 .dockerignore 文件

```
node_modules
.git
.env
*.md
.DS_Store
dist
coverage
```

## 2. 运行时安全

### 2.1 以非 root 用户运行

```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

### 2.2 资源限制

```yaml
# docker-compose.yml
services:
  api:
    image: myapp:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 128M
```

### 2.3 安全扫描

```bash
# Trivy 镜像漏洞扫描
trivy image python:3.12-slim

# Docker Scout 分析
docker scout quickview myapp:latest
```

## 3. Compose 编排规范

### 3.1 完整的 docker-compose.yml 示例

```yaml
version: '3.8'
services:
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    image: docmind-api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/docmind
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./data/uploads:/app/uploads
    networks:
      - app-network

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: docmind
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d docmind"]
      interval: 10s
      retries: 5

  redis:
    image: redis:7.2-alpine
    volumes:
      - redisdata:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api

volumes:
  pgdata:
  redisdata:

networks:
  app-network:
    driver: bridge
```

## 4. 日志管理

### 4.1 应用日志输出到 stdout/stderr

```python
import logging
import sys

# 将日志输出到 stdout，让 Docker 接管
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s'
))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
```

### 4.2 Docker 日志驱动

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 5. 常见问题排查

### 5.1 容器启动即退出

```bash
# 查看退出日志
docker logs <container-id>

# 常见原因：端口冲突、环境变量缺失、依赖服务未就绪
```

### 5.2 内存持续增长（内存泄漏排查）

```bash
docker stats <container-id>     # 实时监控
docker top <container-id>        # 查看进程列表
docker exec <container-id> pmap -x <pid>  # 查看进程内存映射
```

### 5.3 磁盘空间不足

```bash
# 清理未使用的镜像、容器、卷
docker system prune -a --volumes

# 查看磁盘占用
docker system df
```

## 6. CI/CD 集成

### 6.1 GitHub Actions 构建推送示例

```yaml
name: Build and Push Docker Image
on:
  push:
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_TOKEN }}
      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: myapp/docmind:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

> **文档状态**：正式发布 | **维护者**：DevOps 团队 | **更新于**：2025-10-20
