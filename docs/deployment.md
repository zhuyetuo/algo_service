# 部署说明

## 环境要求

| 依赖 | 说明 |
|------|------|
| Docker + Docker Compose | 运行容器 |
| MySQL 8 | 远端服务器 192.168.33.253:30100 |
| TDengine 3.x | 远端服务器 192.168.33.253，REST 端口 6041 |
| 模型文件 | `weights/ml_rf.pkl` + `weights/ml_rf.json`，已随代码库提交，无需训练 |

---

## 首次部署

### 1. 配置环境变量

```bash
cp .env.example .env
```

默认值已对齐远端环境（192.168.33.253:30100），通常无需修改。  
如需覆盖，编辑 `.env` 文件中对应的变量。

### 2. 启动服务

```bash
docker compose up -d --build
```

首次执行会自动构建镜像（约 1–2 分钟），之后启动容器。

### 3. 验证连接

```bash
curl http://localhost:8383/health
```

返回示例：

```json
{"status": "ok", "mysql": "ok", "tdengine": "ok"}
```

如果某个连接失败，对应字段会显示错误信息，`status` 变为 `"degraded"`。

---

## 日常操作

```bash
# 查看实时日志
docker logs algo_service -f

# 查看最近 100 行日志
docker logs algo_service --tail 100

# 查看日志文件（宿主机）
tail -f logs/algo_service.log

# 停止服务（保留容器）
docker compose stop

# 启动已停止的服务
docker compose start

# 重启服务
docker compose restart

# 停止并删除容器（镜像和 volume 保留）
docker compose down

# 强制重新构建镜像并启动
docker compose up -d --build
```

---

## 更新部署

```bash
git pull
docker compose up -d --build
```

---

## 完全清理

```bash
# 删除容器 + 数据卷（model_weights 会被清空，需重新挂载模型）
docker compose down -v

# 同时删除镜像
docker compose down -v --rmi local
```

---

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `docker-compose.yml` | 服务定义，环境变量默认值 |
| `.env.example` | 环境变量模板，复制为 `.env` 后修改 |
| `.env` | 实际生效的配置（不纳入版本库） |
| `logs/` | 日志文件目录（bind mount 到容器内 /app/logs） |
| `weights/` | 模型文件目录（Docker volume） |

---

## 常见问题

**服务启动后立即退出**  
→ 数据库连接失败。检查 `DB_HOST` / `DB_PORT` 是否正确，确认远端 MySQL (192.168.33.253:30100) 可达。

**health 返回 tdengine degraded**  
→ TDengine REST API 不可达。确认 TDengine 容器已启动，端口 6041 已暴露。

**推理周期没有日志输出**  
→ TDengine 中暂无设备数据，或模型文件不存在。确认 `weights/ml_rf.pkl` 存在于容器内 `/app/weights/`。
