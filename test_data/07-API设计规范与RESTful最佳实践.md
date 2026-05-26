# API 设计规范与 RESTful 最佳实践

## 1. URL 设计规范

### 1.1 基本原则

| 规则 | 好例子 | 坏例子 |
|------|--------|--------|
| 使用名词复数 | `/api/v1/documents` | `/api/v1/getDocuments` |
| 层级不超过 3 层 | `/api/v1/kb/{kb_id}/documents/{doc_id}` | `/api/v1/kb/{kb_id}/documents/{doc_id}/chunks/{chunk_id}/metadata` |
| 小写 + 连字符 | `/api/v1/knowledge-bases` | `/api/v1/knowledgeBases` |
| 不用文件扩展名 | `/api/v1/documents` | `/api/v1/documents.json` |

### 1.2 资源嵌套原则

```yaml
# 正确：资源之间的包含关系很明确
GET  /api/v1/knowledge-bases/{kb_id}/documents       # 知识库下的所有文档
POST /api/v1/knowledge-bases/{kb_id}/documents       # 在知识库下创建文档

# 有些资源适合独立访问（通过查询参数）
GET  /api/v1/documents?kb_id={kb_id}                 # 等价于上面的嵌套 URL

# 不要无限嵌套
GET  /api/v1/knowledge-bases/{kb_id}/documents/{doc_id}/chunks/{chunk_id}/annotations
# 如果确实需要深层嵌套，拆分为独立 API
GET  /api/v1/chunks/{chunk_id}/annotations
```

## 2. HTTP 方法与状态码

### 2.1 方法语义

| 方法 | 语义 | 幂等性 | 示例 |
|------|------|--------|------|
| GET | 获取资源 | 是 | `GET /api/v1/documents?status=indexed` |
| POST | 创建资源 | 否 | `POST /api/v1/documents` |
| PUT | 全量替换资源 | 是 | `PUT /api/v1/documents/{id}` |
| PATCH | 部分更新资源 | 否 | `PATCH /api/v1/documents/{id}` (只传要改的字段) |
| DELETE | 删除资源 | 是 | `DELETE /api/v1/documents/{id}` |

### 2.2 状态码使用指南

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/v1/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = await find_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail={
            "error": "document_not_found",
            "message": f"文档 {doc_id} 不存在",
            "doc_id": doc_id
        })
    return doc

# 标准状态码使用场景：
# 200 - 请求成功
# 201 - 资源创建成功（POST 返回值）
# 204 - 请求成功但无返回内容（DELETE 返回值）
# 400 - 请求参数错误（如 kb_id 格式不对）
# 401 - 未认证（Token 缺失或无效）
# 403 - 已认证但无权限（不是你的知识库）
# 404 - 资源不存在
# 409 - 资源冲突（如同时修改同一文档）
# 422 - 请求格式正确但语义错误
# 429 - 请求频率超限
# 500 - 服务器内部错误
# 503 - 服务暂时不可用（如正在重启）
```

## 3. 统一的响应格式

### 3.1 成功响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "documents": [...],
    "total": 156,
    "page": 1,
    "page_size": 20
  },
  "request_id": "req_abc123"
}
```

### 3.2 错误响应

```json
{
  "code": 40001,
  "message": "知识库不存在",
  "detail": "指定的 knowledge_base_id 在系统中未找到",
  "request_id": "req_abc123"
}
```

### 3.3 FastAPI 实现

```python
from pydantic import BaseModel
from typing import Optional, Generic, TypeVar

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: Optional[T] = None
    request_id: str

class PaginatedData(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
```

## 4. 认证与鉴权

### 4.1 API Key 认证

```python
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    token = credentials.credentials
    # 从数据库或 Redis 验证 token
    user = await auth_service.verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return user
```

### 4.2 鉴权中间件

```python
from functools import wraps

def require_kb_access(func):
    """验证用户是否有权限访问指定的知识库"""
    @wraps(func)
    async def wrapper(kb_id: str, user=Depends(verify_api_key), *args, **kwargs):
        has_access = await check_kb_permission(user.id, kb_id)
        if not has_access:
            raise HTTPException(
                status_code=403,
                detail=f"无权访问知识库 {kb_id}"
            )
        return await func(kb_id=kb_id, user=user, *args, **kwargs)
    return wrapper
```

## 5. 分页、排序与筛选

### 5.1 分页参数规范

```
GET /api/v1/documents?page=1&page_size=20&sort_by=created_at&order=desc
```

### 5.2 筛选参数

```
GET /api/v1/documents?
    kb_id=xxx&
    source_type=pdf,markdown&
    status=indexed&
    created_after=2025-01-01&
    created_before=2025-06-01&
    search=Redis
```

### 5.3 实现

```python
from typing import Optional, Literal
from pydantic import BaseModel, Field

class DocumentQuery(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Literal["created_at", "title", "updated_at"] = "created_at"
    order: Literal["asc", "desc"] = "desc"
    source_type: Optional[str] = None  # "pdf,markdown"
    status: Optional[str] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    search: Optional[str] = None
```

## 6. 版本管理

### 6.1 三种策略对比

| 策略 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| URL 版本 | `/api/v1/documents` `/api/v2/documents` | 最直观，浏览器可调试 | URL 变化，客户端需更新 |
| Header 版本 | `Accept: application/vnd.api+json;version=1` | URL 不变 | 调试不方便 |
| 查询参数版本 | `/api/documents?version=1` | 灵活 | 污染 URL |

**推荐方案**：URL 版本（`/api/v1/...`），简单直观，新版本不兼容时直接升级路径。

## 7. 限流设计

### 7.1 令牌桶算法

```python
import time
from collections import defaultdict

class TokenBucketLimiter:
    def __init__(self, rate: int, capacity: int):
        self.rate = rate          # 每秒生成 rate 个 token
        self.capacity = capacity   # 桶的最大容量（允许突发）
        self.tokens = defaultdict(lambda: capacity)
        self.last_check = defaultdict(float)

    def allow(self, key: str) -> bool:
        now = time.time()
        elapsed = now - self.last_check[key]
        self.tokens[key] = min(
            self.capacity,
            self.tokens[key] + elapsed * self.rate
        )
        self.last_check[key] = now

        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True
        return False

# 使用：每用户每秒 10 次请求
limiter = TokenBucketLimiter(rate=10, capacity=20)
```

### 7.2 响应头

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1699123456
Retry-After: 60
```

---

> **文档版本**：v1.2 | **最后更新**：2025-10-20 | **编写**：平台架构组
