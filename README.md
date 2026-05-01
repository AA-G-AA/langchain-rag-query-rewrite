# langchain-rag-query-rewrite

基于 LangChain 1.0 官方 RAG Demo 扩展实现，在官方基础上加入了工程化改进。

## 相比官方 Demo 新增

| 官方 Demo | 本项目 |
|-----------|--------|
| 单次查询检索 | 多查询重写，3个角度提升召回率 |
| 无质量过滤 | Cosine 阈值筛选，过滤低相关文档 |
| 无精排 | BGE Reranker 精排 |
| 单轮对话 | 多轮对话感知，历史上下文影响改写 |
| 无输入校验 | 中间件自动翻译+参数校验，强制英文查询 |

## 环境要求

- Ollama 本地运行
  - `bge-m3:latest`（Embedding）
  - `deepseek-r1:8b`（可替换）
- 云端 OpenAI 兼容接口（生成回答）

## 快速开始

```bash
