# Changelog

## [Unreleased] — 2026-05-26

### Added
- **统一知识库目录** `data/knowledge_base/`，上传和本地文档统一管理
- **`GET /api/v1/documents`** — 列出知识库所有文件
- **`DELETE /api/v1/documents/{filename}`** — 删除文档（同时移除本地文件 + Chroma 向量）
- **索引脚本支持 `--append` 增量模式**（默认全量重建，保证与目录一致）
- **README.md** 功能文档

### Changed
- **上传文件保留原始文件名**存入 `knowledge_base/`，不再用 UUID 重命名
- **索引脚本扫描目录**从 `test_data/` 改为 `knowledge_base/`
- **RAG prompt 强化**：要求先结论再分点、禁编造、无资料时明确告知
- **config.py** 新增 `KNOWLEDGE_BASE_DIR` 配置项

### Fixed
- **TextLoader 编码**：默认 GBK → UTF-8，修复中文 txt 文件加载失败
- **start-frontend.bat**：`;` 语法错误改为 `&&`（Windows CMD 不兼容分号）
- **知识库文件管理混乱**：上传目录和测试目录分离导致索引不一致

### Technical Debt
- 前端 Turbopack 开发服务器缓存不定期损坏，暂时改用 `next build && next start` 生产模式
