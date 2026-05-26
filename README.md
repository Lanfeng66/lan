# DocMind - 企业级多源技术文档智能问答系统

基于 LangGraph + RAG 的智能文档问答助手，支持多格式文档、多轮对话、流式输出。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM 编排 | LangChain + LangGraph |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-m3（本地 GPU/CPU，FP16 半精度） |
| 检索策略 | MMR 向量检索 + BM25 关键词 + RRF 融合 + LLM Rerank |
| 后端框架 | FastAPI + SSE 流式输出 |
| 前端 | Next.js 16 + React 19 + Tailwind CSS v4 |
| 评估体系 | RAGAS（忠实度、相关性、上下文精确度/召回率） |

## 快速启动

### 1. 环境准备

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. 配置环境变量

项目根目录下 `.env`：

```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-your-key
LLM_MODEL=deepseek-chat
CHROMA_PERSIST_DIR=./data/chroma_db
KNOWLEDGE_BASE_DIR=./data/knowledge_base
```

### 3. 准备文档 & 索引

将文档放入 `data/knowledge_base/`，然后：

```bash
cd backend
python scripts/index_documents.py
```

### 4. 一键启动

- 双击 `start-backend.bat` → 后端 :8000
- 双击 `start-frontend.bat` → 前端 :3000
- 双击 `stop.bat` → 关闭所有服务

浏览器打开 `http://localhost:3000`，前端自动跳转到聊天界面。

## 功能特性

### 核心 RAG 管线（Fast 模式，默认）

- **Query 改写**：口语化问题 → 检索关键词
- **MMR 向量检索**：语义搜索 + 多样性平衡
- **BM25 关键词检索**：jieba 分词 + 关键词匹配
- **RRF 融合**：双路检索结果融合去重
- **LLM Rerank**：批量并行打分，精排 Top-5
- **流式生成**：SSE 逐字输出，打字机效果

> 可通过 `.env` 设置 `RAG_MODE=quality` 开启幻觉检测 + 自修正回路（会额外增加 10+ 次 LLM 调用，仅建议在答案质量要求极高时使用）。

### 文档管理

- **POST /documents/upload** — 上传文档，自动 加载→分割→索引
- **DELETE /documents/{filename}** — 删除文档，同步移除文件 + 向量
- **GET /documents** — 列出所有已上传文档
- 所有文档统一存入 `data/knowledge_base/` 目录

### 多轮对话

- 指代消解（"它怎么部署？" → 自动补全上下文）
- 话题切换检测
- 对话摘要压缩

### 前端交互

- 流式打字机效果（SSE）
- 答案引用来源标注
- 检索调试面板（`/debug`）：可视化每一步检索过程

### Embedding GPU 加速

- 自动检测 CUDA / MPS / CPU
- GPU 启用 FP16 半精度，显存减半
- 批量编码 98 chunks 仅 0.6s

## 项目结构

```
DocMind/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 入口
│   │   ├── api/
│   │   │   ├── chat.py             # 问答 + 流式 SSE
│   │   │   ├── documents.py        # 文档上传/删除/列举
│   │   │   ├── feedback.py         # 用户反馈
│   │   │   ├── knowledge_bases.py  # 知识库 CRUD
│   │   │   └── debug.py            # 检索调试
│   │   ├── core/
│   │   │   └── auth.py             # API Key 认证
│   │   ├── ingestion/
│   │   │   ├── loader.py           # 多格式文档加载
│   │   │   ├── splitter.py         # 文本分割
│   │   │   └── github_source.py    # GitHub 数据源
│   │   ├── embedding/
│   │   │   └── embeddings.py       # Embedding 模型（GPU/FP16）
│   │   ├── storage/
│   │   │   └── vector_store.py     # ChromaDB 操作（批量索引/MMR）
│   │   ├── llm/
│   │   │   └── chat_model.py       # LLM 封装
│   │   ├── graph/
│   │   │   ├── state.py            # LangGraph 状态
│   │   │   ├── nodes.py            # 图节点（检索/融合/重排/生成）
│   │   │   └── workflow.py         # 工作流组装（fast/quality 双模式）
│   │   ├── conversation/
│   │   │   └── manager.py          # 多轮对话管理
│   │   └── quality/
│   │       └── hallucination_check.py  # 幻觉检测
│   └── scripts/
│       ├── index_documents.py      # 文档索引
│       ├── evaluate.py             # RAGAS 评估
│       └── sync_github.py          # GitHub 同步
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # 首页导航
│   │   ├── chat/
│   │   │   └── page.tsx            # 聊天界面（流式 SSE）
│   │   ├── debug/
│   │   │   └── page.tsx            # 检索调试面板
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── package.json
├── data/
│   ├── knowledge_base/             # 统一文档管理目录
│   ├── uploads/                    # 旧上传目录（保留）
│   └── chroma_db/                  # 向量持久化
├── test_data/                      # 原始测试文档
├── config.py                       # 中心化配置
├── .env                            # 环境变量（不提交 git）
├── start-backend.bat               # 启动后端
├── start-frontend.bat              # 启动前端
└── stop.bat / stop.py              # 关闭服务
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 单轮/多轮问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| POST | `/api/v1/documents/upload` | 上传文档并索引 |
| GET | `/api/v1/documents` | 列出知识库所有文件 |
| DELETE | `/api/v1/documents/{filename}` | 删除文档（文件+向量） |
| POST | `/api/v1/feedback` | 提交反馈 |
| GET | `/api/v1/feedback/stats` | 反馈统计 |
| GET | `/api/v1/debug/retrieve` | 检索调试 |
| GET | `/health` | 健康检查 |

## RAGAS 评估

```bash
python backend/scripts/evaluate.py
```

评估指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
