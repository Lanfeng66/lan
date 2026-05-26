# DocMind - 企业级多源技术文档智能问答系统

基于 LangGraph + RAG 的智能文档问答助手，支持多格式文档、多轮对话、流式输出、幻觉检测与自修正。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM 编排 | LangChain + LangGraph |
| 向量数据库 | ChromaDB |
| Embedding | BAAI/bge-m3 |
| 检索策略 | 混合检索（向量 + BM25）+ RRF 融合 + Cross-encoder 重排序 |
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
```

### 3. 索引文档

```bash
cd D:\PythonProject\DocMind
python backend/scripts/index_documents.py
```

### 4. 启动后端

```bash
cd D:\PythonProject\DocMind
python -m uvicorn backend.app.main:app --reload --port 8000
```

### 5. 启动前端

```bash
cd frontend
npm install
npx next build; npx next start
```

浏览器打开 `http://localhost:3000`。

## 功能特性

### 核心 RAG 管线

- **Query 改写**：口语化问题自动转为检索关键词
- **混合检索**：向量语义检索 + BM25 关键词检索，RRF 融合排序
- **Cross-encoder 重排序**：对候选文档精排，取 Top-5
- **Query 分解**：复杂对比类问题自动拆分为子问题分别检索
- **幻觉检测**：NLI 检查答案中的事实断言是否有上下文依据
- **自修正回路**：幻觉率过高时自动重新生成

### 多轮对话

- 指代消解（"它怎么部署？" → 自动补全上下文）
- 话题切换检测
- 对话摘要压缩（长对话自动压缩早期消息）

### 前端交互

- 流式打字机效果（SSE）
- 答案引用来源标注
- 点赞/点踩反馈闭环

### 管理功能

- 知识库 CRUD API
- 检索调试面板（`/debug`）：可视化每一步检索过程
- GitHub 仓库增量同步
- API Key 认证

## 项目结构

```
DocMind/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI 入口
│   │   ├── api/
│   │   │   ├── chat.py             # 问答 + 流式 SSE
│   │   │   ├── documents.py        # 文档上传管理
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
│   │   │   └── embeddings.py       # Embedding 模型
│   │   ├── storage/
│   │   │   └── vector_store.py     # ChromaDB 操作
│   │   ├── llm/
│   │   │   └── chat_model.py       # LLM 封装
│   │   ├── graph/
│   │   │   ├── state.py            # LangGraph 状态
│   │   │   ├── nodes.py            # 图节点（含幻觉检测、自修正）
│   │   │   └── workflow.py         # 工作流组装
│   │   ├── conversation/
│   │   │   └── manager.py          # 多轮对话管理
│   │   └── quality/
│   │       └── hallucination_check.py  # 幻觉检测模块
│   └── scripts/
│       ├── index_documents.py      # 文档索引
│       ├── evaluate.py             # RAGAS 评估
│       ├── sync_github.py          # GitHub 增量同步
│       └── generate_api_key.py     # API Key 生成
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # 对话主界面
│   │   ├── debug/
│   │   │   └── page.tsx            # 检索调试面板
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── package.json
├── test_data/                      # 测试文档
├── data/                           # 向量数据、上传文件
└── .env
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 单轮/多轮问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| POST | `/api/v1/documents/upload` | 上传文档并索引 |
| DELETE | `/api/v1/documents/{id}` | 删除文档 |
| POST | `/api/v1/feedback` | 提交反馈 |
| GET | `/api/v1/feedback/stats` | 反馈统计 |
| POST | `/api/v1/knowledge-bases` | 创建知识库 |
| GET | `/api/v1/knowledge-bases` | 列出知识库 |
| DELETE | `/api/v1/knowledge-bases/{id}` | 删除知识库 |
| GET | `/api/v1/debug/retrieve` | 检索调试 |
| GET | `/health` | 健康检查 |

## GitHub 增量同步

```bash
python backend/scripts/sync_github.py --repo https://github.com/用户/仓库.git --branch main
```

首次运行克隆全量索引，再次运行仅同步变更文件。

## 生成 API Key

```bash
python backend/scripts/generate_api_key.py
```

将生成的 Key 添加到 `.env` 的 `API_KEYS` 中，请求时带 `Authorization: Bearer <key>`。

## RAGAS 评估

```bash
python backend/scripts/evaluate.py
```

评估指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall。
