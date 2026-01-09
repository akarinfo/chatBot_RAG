# 项目上下文压缩（给助手/新接手者）

## 项目定位
- RAG 知识库问答：文档 → 向量库（Weaviate）→ 检索 → LLM 生成
- 模型均为 **API 调用**，不在本地跑模型

## 技术栈
- LLM：DeepSeek（OpenAI 兼容）
- Embedding：ModelScope（OpenAI 兼容）
- Vector DB：Weaviate（docker-compose）
- Web：Streamlit
- 核心依赖：`langchain`, `langgraph`, `langchain-openai`, `langchain-community`

## 关键文件
- `app.py`：Streamlit UI（聊天 + 文档管理 + 重新入库）
- `src/services/ingest/processor.py`：入库流程（Markdown 结构分块 + 递归分块 → Weaviate）
- `src/workflows/rag_bot/graph.py`：RAG 逻辑（retriever.invoke → LLM）
- `src/core/llm.py`：LLM/Embedding 客户端
- `src/core/vectordb.py`：Weaviate 客户端与检索器
- `docker-compose.yml`：本地 Weaviate 服务
- `docs/PROJECT_GUIDE.zh-CN.md`：新手说明 + 流程图

## 运行流程（最短路径）
1) `docker compose up -d`
2) `.env` 配置 API keys + 模型名
3) `PYTHONPATH=src python -m services.ingest.processor`
4) `streamlit run app.py`

## 必填环境变量（不写明文）
- LLM：`LLM_PROVIDER=deepseek`, `DEEPSEEK_API_KEY`
- Embedding：`EMBED_PROVIDER=modelscope`, `MODELSCOPE_API_TOKEN`, `MODELSCOPE_EMBED_MODEL`
- Weaviate：`WEAVIATE_URL`（端口跟 compose 映射一致）

## 重要约束
- Embedding 模型与入库必须一致（改模型需重新入库）
- Weaviate ready 接口返回 200 且 body 为空属正常

## 常见坑
- 端口冲突（已有 weaviate 占用 8080/8081）
- ModelScope token 误用在 DashScope（会 401）
- retriever API 变动（用 `retriever.invoke()`）
