# 项目说明（新手上手版）

本项目是一个 **RAG 知识库问答机器人**，采用：
- **LLM**：DeepSeek（OpenAI 兼容接口）
- **Embedding**：ModelScope（OpenAI 兼容接口）
- **向量数据库**：Weaviate
- **前端**：Streamlit（带知识库文档管理）

目标是：把 `data/` 里的文档入库为向量 → 通过检索增强生成回答。

---

## 目录结构

- `app.py`：Streamlit Web 应用（聊天 + 文档管理）
- `src/services/ingest/processor.py`：入库脚本（Markdown 结构化分块 + 向量写入 Weaviate）
- `src/workflows/rag_bot/graph.py`：RAG 逻辑（检索 → 生成）
- `src/core/llm.py`：统一管理 LLM/Embedding
- `src/core/vectordb.py`：Weaviate 客户端与检索器
- `data/`：知识库原始文档
- `docker-compose.yml`：Weaviate 本地服务
- `.env`：你的 API key（不要提交）

---

## 快速启动

1) 启动 Weaviate
```bash
docker compose up -d
```
健康检查（返回 200 即可）：
```bash
curl http://localhost:8081/v1/.well-known/ready
```

2) 配置 `.env`（示例在 `.env.example`）
```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeekKey

EMBED_PROVIDER=modelscope
MODELSCOPE_API_TOKEN=你的ModelScopeToken
MODELSCOPE_EMBED_MODEL=Qwen/Qwen3-Embedding-8B

WEAVIATE_URL=http://localhost:8081
```

3) 入库
```bash
PYTHONPATH=src python -m services.ingest.processor
```

4) 启动 Web
```bash
streamlit run app.py
```

---

## 核心流程说明

### 1. 入库（ingest）
入口：`src/services/ingest/processor.py`

流程：
1) 读取 `data/` 下 `.md/.mdx/.txt`
2) Markdown 先按标题结构拆分，再做递归分块
3) 调用 ModelScope embeddings 生成向量
4) 写入 Weaviate 的 `RAGChunk` 类

> 注意：Embedding 模型一旦入库确定，检索时必须使用同一模型。

### 2. 检索 + 生成（RAG）
入口：`src/workflows/rag_bot/graph.py`

流程：
1) 通过 Weaviate 检索相关 chunk（MMR 策略）
2) 拼接上下文 + 提示词
3) 调用 DeepSeek 生成回答（只允许基于上下文）

### 3. Web 界面
入口：`app.py`

功能：
- 上传/删除 `data/` 文件
- 一键重新入库
- 聊天问答（流式输出）

---

## 流程图（Mermaid）

```mermaid
flowchart TD
    A[用户文档 data/] --> B[services/ingest/processor.py<br/>Markdown 结构化分块 + 递归分块]
    B --> C[ModelScope Embedding API]
    C --> D[Weaviate 向量库 RAGChunk]

    E[用户提问] --> F[workflows/rag_bot/graph.py<br/>Retriever.invoke()]
    F --> D
    F --> G[拼接上下文 + Prompt]
    G --> H[DeepSeek LLM API]
    H --> I[回答输出]

    J[Streamlit UI] --> E
    J --> B
```

---

## 常见问题排查

1) **Weaviate 无响应**
- 确认端口：如果 compose 映射到 8081，就访问 `http://localhost:8081`
- 运行 `docker logs -f weaviate` 看是否启动成功

2) **Embedding 401/无效**
- 确认 `.env` 中的 `MODELSCOPE_API_TOKEN` 是否正确
- 确保 `MODELSCOPE_EMBED_MODEL` 已填写

3) **回答质量差**
- 确保文档内容里真的包含答案
- 调整分块大小（`services/ingest/processor.py`）
- 增加 `k` 值（`core/vectordb.py` 中 `search_kwargs`）

---

## 开发建议

- 新增文档后，必须重新运行 `PYTHONPATH=src python -m services.ingest.processor`
- `.env` 不要提交到 Git
- 如果需要多租户/多知识库，可扩展 Weaviate class 或加命名空间
