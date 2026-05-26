# Redis 集群运维手册

## 第一章：集群架构概述

### 1.1 什么是 Redis Cluster？

Redis Cluster 是 Redis 官方提供的分布式解决方案，支持数据自动分片（sharding）、故障转移（failover）和在线扩缩容。它将整个数据集按 slot 划分到多个节点上，共有 16384 个哈希槽。

### 1.2 集群拓扑要求

- 至少 6 个 Redis 实例（3 主 + 3 从）才能组成高可用集群
- 每个主节点至少配 1 个从节点，否则主节点故障时集群不可用
- 所有节点之间通过 TCP 长连接通信，需开放端口 6379（客户端）和 16379（集群总线）

### 1.3 Slot 分配机制

Redis Cluster 使用 CRC16(key) % 16384 计算 key 所属的 slot。常见的 slot 分配策略：

```
节点 A: slots 0-5460     (5461 个槽)
节点 B: slots 5461-10922 (5462 个槽)
节点 C: slots 10923-16383 (5461 个槽)
```

### 1.4 MOVED 与 ASK 重定向

当客户端请求的 key 不在当前节点时：
- **MOVED**：永久重定向，key 确实迁移到了目标节点，客户端应更新本地 slot 表
- **ASK**：临时重定向，slot 正在迁移中，仅本次请求转向目标节点

## 第二章：集群部署

### 2.1 环境准备

```bash
# 安装 Redis 7.2.x
wget https://download.redis.io/releases/redis-7.2.4.tar.gz
tar xzf redis-7.2.4.tar.gz
cd redis-7.2.4 && make -j$(nproc)
```

### 2.2 配置文件模板

每个节点需要一个 redis.conf，关键配置如下：

```conf
port 6379
cluster-enabled yes
cluster-config-file nodes-6379.conf
cluster-node-timeout 15000
appendonly yes
maxmemory 4gb
maxmemory-policy allkeys-lru
```

### 2.3 创建集群

```bash
redis-cli --cluster create \
  192.168.1.10:6379 192.168.1.11:6379 192.168.1.12:6379 \
  192.168.1.20:6380 192.168.1.21:6380 192.168.1.22:6380 \
  --cluster-replicas 1
```

## 第三章：集群扩容

### 3.1 垂直扩容（增加内存）

单机 maxmemory 调大需要重启，对于有从节点的场景，可以先调大从节点 maxmemory → 主从切换 → 调大原主节点 maxmemory。

### 3.2 水平扩容（增加节点）

```bash
# 步骤一：添加新节点到集群
redis-cli --cluster add-node 192.168.1.13:6379 192.168.1.10:6379

# 步骤二：重新分配 slot
redis-cli --cluster reshard 192.168.1.10:6379 \
  --cluster-from all \
  --cluster-to <new-node-id> \
  --cluster-slots 4096

# 步骤三：为新节点添加从节点
redis-cli --cluster add-node 192.168.1.23:6380 192.168.1.13:6379 \
  --cluster-slave --cluster-master-id <new-master-id>
```

### 3.3 扩容期间的注意事项

1. **迁移过程中集群仍可对外服务**，但迁移中的 slot 会有短暂不可用
2. **迁移速度控制**：使用 `cluster-migration-barrier` 参数控制每小时迁移的 slot 数
3. **禁止批量迁移**：同时迁移太多 slot 会导致集群不稳定
4. **建议在业务低峰期执行**，迁移单个 slot 的时间取决于该 slot 中的数据量

### 3.4 常见扩容问题

**Q: 迁移过程中某个 key 的迁移失败怎么办？**
使用 `redis-cli --cluster fix` 重试，或手动 `MIGRATE` 命令单个 key。

**Q: 扩容后客户端报大量 MOVED 错误？**
客户端库需要支持集群模式（如 JedisCluster、redis-py-cluster），自动处理 MOVED 重定向。

## 第四章：故障处理

### 4.1 常见故障场景

| 故障类型 | 现象 | 恢复方案 |
|----------|------|----------|
| 从节点宕机 | 集群正常运行，无数据丢失 | 重启从节点即可，在从节点恢复前主节点压力增大 |
| 主节点宕机 | 从节点自动提升为主，集群短暂不可用（cluster-node-timeout 窗口内） | 无需人工干预，自动 failover |
| 脑裂（Split-Brain） | 网络分区导致两个主节点同时服务相同 slot | 需人工介入，选择数据较新的一侧恢复 |
| 集群完全不可用 | cluster-require-full-coverage=yes 且某个 slot 无节点负责 | 紧急将 cluster-require-full-coverage 改为 no，先恢复服务 |

### 4.2 手动 Failover

```bash
# 在从节点上执行，将当前从节点提升为主节点
redis-cli -p 6380 cluster failover

# 强制模式（主节点不可达时使用）
redis-cli -p 6380 cluster failover force
```

## 第五章：监控指标

### 5.1 关键指标

```bash
# 集群状态检查
redis-cli cluster info

# 各节点状态
redis-cli cluster nodes

# 查看 key 的 slot 分布
redis-cli --cluster check 192.168.1.10:6379
```

### 5.2 Prometheus 监控指标

| 指标名称 | 含义 | 告警阈值 |
|----------|------|----------|
| redis_cluster_slots_fail | 故障 slot 数 | > 0 |
| redis_connected_slaves | 已连接从节点数 | < 1（主节点）|
| redis_db_keys | 当前 key 总数 | 仅做监控趋势 |
| redis_memory_used_bytes / redis_memory_max_bytes | 内存使用率 | > 80% |
| redis_rejected_connections_total | 拒绝连接数 | > 0 |
| instantaneous_ops_per_sec | 每秒操作数 | 仅做监控趋势 |

### 5.3 慢查询监控

```bash
# 设置慢查询阈值（微秒）
CONFIG SET slowlog-log-slower-than 10000

# 查看最近的慢查询
SLOWLOG GET 10
```

## 第六章：数据备份与恢复

### 6.1 RDB 备份

```bash
# 手动触发 RDB 快照
redis-cli BGSAVE

# 备份策略配置
save 900 1      # 900秒内至少1个key变化
save 300 10     # 300秒内至少10个key变化
save 60 10000   # 60秒内至少10000个key变化
```

### 6.2 AOF 备份

AOF（Append Only File）记录每次写操作，数据安全性高于 RDB 但文件更大。

```conf
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 6.3 灾备恢复流程

1. 停止所有节点写操作
2. 确认最新备份文件完整性（校验 SHA256）
3. 清空目标节点数据目录
4. 复制备份文件到数据目录
5. 逐个启动节点，等待集群状态恢复
6. 验证数据完整性（抽样 key 比对）

---

> **文档版本**：v2.3 | **最后更新**：2025-11-15 | **维护团队**：基础架构组
