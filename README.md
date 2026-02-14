# 小说可视化生成器

一个基于 Streamlit 的可视化小说生成器，支持：

- 世界观动态创建与编辑（可用提示词 + LLM 生成 + 提示词改写）
- 角色与场景管理（支持 LLM JSON 生成 + 手工编辑 + 删除）
- 事件发展线与节点管理（增删改、节点顺序调整、可视化）
- 全故事线连贯生成 + 单节点重生成（基于历史脉络）
- 导出完整小说 Markdown

## 快速开始

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

> 直接执行 `python app.py` 会提示正确启动方式，避免 bare mode 的 ScriptRunContext 警告。

## LLM 调用

应用内部集成了你提供的 `get_chat_response` 形式，并支持：

- OpenAI 兼容接口（`model` 不含 `ollama`）
- Ollama（`model` 传入形如 `ollama#qwen2.5:7b`）

## 数据存储

- 业务数据：`data/novel_data.json`
- token 日志：`data/token_log.jsonl`
