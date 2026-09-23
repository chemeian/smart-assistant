# Smart Assistant · 数据分析 Agent

![CI](https://github.com/chemeian/smart-assistant/actions/workflows/ci.yml/badge.svg)

一个基于 Flask 的 **AI 数据分析助手**：上传 CSV，Agent 自动读数据、画图、统计、写结论。

> ![数据分析演示](docs/demo1.png)

![分析结果](docs/demo2.png)

---

## 它能做什么

| 场景 | 效果 |
|---|---|
| 📊 数据分析 | 上传 CSV/Excel，自动出直方图、相关性热力图，给中文结论 |
| 💬 智能对话 | 对接 DeepSeek / 通义千问，流式输出，多轮上下文 |
| 🤖 Agent 模式 | Function Calling 自动调工具，Planner 拆任务，权限确认 |
| 📝 NLP | 情感判别、关键词、摘要、实体识别 |
| 📥 导出 | 一键把对话存成 Markdown |

---

## 技术栈

- **后端**：Flask（工厂模式 + 蓝图分层）
- **大模型**：DeepSeek / 通义千问（OpenAI 兼容接口，SSE 真逐 token 流式）
- **Agent**：Function Calling、ReAct 循环、Planner、历史摘要
- **数据**：pandas、numpy、matplotlib（Agg 无头后端）
- **存储**：SQLite 持久化对话
- **工程**：pytest 测试、GitHub Actions CI、Docker

---

## 项目结构

```
smart-assistant/
├── app.py              # Flask 工厂入口
├── config.py          # 配置
├── routes/            # 路由层（接请求）
│   ├── chat.py
│   ├── chart.py
│   ├── nlp.py
│   └── agent.py
├── services/          # 业务逻辑层
│   ├── llm_service.py
│   ├── chart_service.py
│   ├── nlp_service.py
│   ├── agent.py       # Agent 核心
│   └── db.py
├── utils/             # 工具函数
├── templates/
│   └── index.html
├── tests/             # pytest
└── .github/workflows/ # CI
```

---

## 快速运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
python run.py
```

访问 <http://127.0.0.1:5000/>
