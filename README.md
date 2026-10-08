# AI 智能伴侣 · Agent 版

一个基于 **ReAct 范式** 与 **Function Calling** 从零构建的 AI Agent 智能体应用，具备自主思考、规划行动与工具调用能力。可跨会话记住用户偏好，集成 DeepSeek 大模型与高德地图 API，实现真实数据调用。


## ✨ 核心能力

| 能力 | 说明 |
|---|---|
| 🧠 Agent 核心引擎 | 基于 ReAct「思考-行动-观察」内层循环，大模型能根据问题自主规划、连续调用工具，直至生成最终回答 |
| 🔧 Function Calling 工具系统 | 自研工具注册表 + 8 个工具：时间查询、文件读写、长期记忆保存/检索、天气查询、地理编码、路径规划 |
| 🧠 长期记忆 + RAG | 跨会话持久化用户偏好，按语义相关度召回最相关记忆，解决大模型无状态与上下文膨胀问题 |
| 📦 上下文自动压缩 | 对话超阈值自动转摘要，防止 token 溢出 |
| 🛡️ 死循环防护 | 最大思考步数限制，避免无限循环、控制调用成本 |
| 📍 真实数据调用 | 集成 DeepSeek 大模型与高德地图 API（天气、路线） |

## 🛠️ 技术栈

Python · Streamlit · DeepSeek API · Function Calling · ReAct · RAG · 高德地图 API · LRU

## 📁 项目结构

```
├── ai_partner_agent.py      # 主程序（Agent 版）
├── requirements.txt         # 依赖
├── sessions/                # 会话记录（自动生成）
├── long_memory_db.json      # 长期记忆库（自动生成）
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install streamlit openai
```

### 2. 配置环境变量

需要两个 API Key：

| 变量 | 用途 | 获取方式 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 大模型调用 | https://platform.deepseek.com |
| `AMAP_API_KEY` | 天气/地图 | https://lbs.amap.com （选「Web 服务」类型） |

Windows 设置：

```bat
set DEEPSEEK_API_KEY=你的key
set AMAP_API_KEY=你的key
```

> 💡 程序会自动回退读取 Windows 系统环境变量，无需重复设置。

### 3. 启动

```bash
streamlit run ai_partner_agent.py
```

## 💬 使用示例

- 「现在几点了」→ 触发时间工具
- 「杭州今天天气怎么样」→ 触发天气工具（高德 API）
- 「从杭州东站到西湖怎么去」→ 触发路径规划工具
- 「记住我喜欢喝冰美式」→ 保存长期记忆
- 新开对话问「我喜欢喝什么」→ 检索长期记忆

## 🔧 工具列表

- `get_current_time`：查询当前时间
- `read_file` / `write_file`：读写本地文件
- `save_long_memory`：保存长期记忆
- `query_long_memory`：检索长期记忆（轻量 RAG）
- `get_weather`：查询天气（高德）
- `geocode`：地理编码（高德）
- `route_plan`：路径规划，驾车/步行/公交（高德）

## 📌 说明

- 本项目为学习 / 个人项目，用于演示 Agent 底层实现逻辑
- 长期记忆采用轻量词频检索（零额外依赖），可替换为 sentence-transformers 语义检索
