# MySQL 性能调优实战

## 第一章：索引优化

### 1.1 B+Tree 索引原理

MySQL InnoDB 使用 B+Tree 作为索引结构。理解 B+Tree 对写出高效 SQL 至关重要：

- **叶子节点**存储所有数据，并通过双向链表连接，支持范围查询
- **非叶子节点**只存储键值和子节点指针，树的高度通常为 2-4 层
- **聚簇索引**（主键索引）的叶子节点存储完整行数据
- **二级索引**的叶子节点存储主键值，查询需要**回表**

### 1.2 联合索引的最左前缀原则

```sql
-- 创建联合索引
CREATE INDEX idx_user_status_time ON orders(user_id, status, created_at);

-- 命中索引（使用了最左列 user_id）
SELECT * FROM orders WHERE user_id = 12345;

-- 命中索引（user_id + status 都使用到）
SELECT * FROM orders WHERE user_id = 12345 AND status = 'paid';

-- 命中索引（user_id + status + created_at 全部使用到）
SELECT * FROM orders
WHERE user_id = 12345 AND status = 'paid' AND created_at > '2025-01-01';

-- 不命中！跳过了 status，只能用 user_id
SELECT * FROM orders WHERE user_id = 12345 AND created_at > '2025-01-01';

-- 不命中！范围查询 user_id 后的 status 失效
SELECT * FROM orders
WHERE user_id > 10000 AND status = 'paid' AND created_at > '2025-01-01';
```

### 1.3 覆盖索引（Covering Index）

覆盖索引是指查询列和条件列都在同一个索引中，避免回表操作：

```sql
-- 需要回表：索引只有 user_id，需要回表查 name, email
SELECT name, email FROM users WHERE user_id = 12345;

-- 覆盖索引：不需要回表，Extra 中显示 Using index
CREATE INDEX idx_user_cover ON users(user_id, name, email);
SELECT name, email FROM users WHERE user_id = 12345;
```

在 RAG 项目中，文档元数据筛选经常用到覆盖索引：

```sql
-- 知识库文档查询优化
CREATE INDEX idx_doc_kb_status ON documents(kb_id, status, title, created_at);
```

### 1.4 索引失效场景

| 场景 | 原因 | 解决方案 |
|------|------|----------|
| `WHERE func(column) = 'value'` | 函数包裹列导致索引失效 | 改为 `WHERE column = ...` 或在生成列上建索引 |
| `WHERE column LIKE '%value'` | 前导模糊查询 | 改用全文索引或 Elasticsearch |
| `WHERE column != 'value'` | 否定条件通常不用索引 | 改为 `IN` 或重新设计条件 |
| 隐式类型转换 | `WHERE varchar_col = 123` 会转为 `WHERE CAST(varchar_col AS SIGNED) = 123` | 参数类型与列类型一致 |
| JOIN 字段字符集不同 | 字符集转换触发全表扫描 | 统一字符集为 utf8mb4 |

## 第二章：查询优化

### 2.1 EXPLAIN 解读

```sql
EXPLAIN SELECT o.*, u.name
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.created_at > '2025-01-01' AND o.status = 'paid';
```

关键字段说明：
- **type**（从好到差）：system > const > eq_ref > ref > range > index > ALL
  - `range` 及以上是可接受的
  - `ALL` 表示全表扫描，必须优化
- **key**：实际使用的索引名称
- **rows**：估计扫描行数，越小越好
- **Extra**：额外信息
  - `Using index` → 覆盖索引，最佳
  - `Using filesort` → 需要额外排序，可能需要优化
  - `Using temporary` → 使用了临时表，需要关注
  - `Using where` → 在存储引擎层过滤

### 2.2 分页优化

```sql
-- 低效：OFFSET 大时扫描大量无用行
SELECT * FROM documents WHERE kb_id = 'xxx' ORDER BY created_at DESC
LIMIT 10000, 20;

-- 优化方案 1：基于游标的分页（推荐）
SELECT * FROM documents WHERE kb_id = 'xxx'
  AND created_at < '2025-03-01 12:00:00'  -- 上一页最后一条的时间
ORDER BY created_at DESC
LIMIT 20;

-- 优化方案 2：先取 ID 再关联（适用于无删除场景）
SELECT d.* FROM documents d
JOIN (
    SELECT id FROM documents
    WHERE kb_id = 'xxx'
    ORDER BY created_at DESC LIMIT 10000, 20
) t ON d.id = t.id;
```

### 2.3 JOIN 优化

```sql
-- 驱动表原则：用小结果集驱动大结果集
-- 错误：大表驱动小表
SELECT * FROM big_table b
JOIN small_table s ON b.sid = s.id;

-- 正确：小表驱动大表
SELECT * FROM small_table s
JOIN big_table b ON s.id = b.sid;

-- NLJ（Nested Loop Join）vs BNL（Block Nested Loop）
-- 确保被驱动表的 JOIN 列有索引，否则退化到 BNL
```

## 第三章：Schema 设计

### 3.1 字段类型选择

| 场景 | 推荐类型 | 原因 |
|------|----------|------|
| 主键/ID | BIGINT UNSIGNED | 空间够用（2^64），比 UUID 省空间 |
| 状态字段 | TINYINT（不要用 ENUM） | ENUM 改值需要 ALTER TABLE |
| JSON 数据 | JSON 类型（MySQL 8.0+） | 支持 JSON 函数、虚拟列索引 |
| 时间戳 | TIMESTAMP(3) 或 DATETIME(3) | 毫秒精度，用于 RAG 中的对话时间线 |
| 长文本 | MEDIUMTEXT | TEXT 最大 64KB，MEDIUMTEXT 最大 16MB |
| 布尔值 | TINYINT(1) | MySQL 没有真正的 BOOLEAN 类型 |

### 3.2 分区表

对于文档 chunk 这样的大表：

```sql
CREATE TABLE chunks (
    id BIGINT UNSIGNED NOT NULL,
    kb_id CHAR(36) NOT NULL,
    doc_id CHAR(36) NOT NULL,
    chunk_index INT NOT NULL,
    content MEDIUMTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at),
    INDEX idx_kb (kb_id, doc_id),
    INDEX idx_doc (doc_id, chunk_index)
) PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p2026 VALUES LESS THAN (2027),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);
```

## 第四章：事务与锁

### 4.1 隔离级别选择

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 适用场景 |
|----------|------|-----------|------|----------|
| READ UNCOMMITTED | 是 | 是 | 是 | 几乎不用 |
| READ COMMITTED | 否 | 是 | 是 | 多数场景，PostgreSQL 默认 |
| REPEATABLE READ | 否 | 否 | 部分 | MySQL 默认，Gap Lock 防幻读 |
| SERIALIZABLE | 否 | 否 | 否 | 最强但几乎不用（性能极差） |

```sql
-- 查看当前隔离级别
SELECT @@transaction_isolation;

-- RAG 项目中，文档元数据读取建议 RC 级别即可
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

### 4.2 死锁排查

```sql
-- 查看当前锁等待情况
SELECT * FROM information_schema.innodb_locks;
SELECT * FROM information_schema.innodb_lock_waits;

-- 查看最近一次死锁详情
SHOW ENGINE INNODB STATUS\G
-- 查看 LATEST DETECTED DEADLOCK 部分
```

## 第五章：监控与巡检

### 5.1 慢查询日志

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 0.5;  -- 500 毫秒即记录
SET GLOBAL log_queries_not_using_indexes = ON;
```

### 5.2 关键性能指标

```sql
-- 连接数
SHOW STATUS LIKE 'Threads_connected';

-- 查询缓存命中率（MySQL 8.0 已移除查询缓存）
-- 替代方案：应用层 Redis 缓存

-- InnoDB 缓冲池命中率
SELECT
  (1 - (SUM(innodb_buffer_pool_reads) / SUM(innodb_buffer_pool_read_requests))) * 100
  AS buffer_pool_hit_rate
FROM information_schema.innodb_buffer_pool_stats;
-- 目标：> 99.9%
```

---

> **文档版本**：v3.2 | **适用版本**：MySQL 8.0+ | **编写**：DBA 团队
