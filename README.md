# Smart Assistant · 智能助手平台

![CI](https://github.com/chemeian/smart-assistant/actions/workflows/ci.yml/badge.svg)

基于 Flask 的一站式 AI 助手，集成**智能对话、数据分析可视化、Agent 智能体、NLP 文本处理**四大能力。

![智能对话](https://raw.githubusercontent.com/chemeian/smart-assistant/main/docs/screen1.png)

![数据分析](https://raw.githubusercontent.com/chemeian/smart-assistant/main/docs/screen2.png)

## 功能

| 模块 | 能力 |
|---|---|
| 💬 智能对话 | DeepSeek / 通义千问双模型、SSE 流式输出、多轮上下文、图片提问、失败自动降级 |
| 🤖 Agent 智能体 | Function Calling、ReAct 循环、Planner 拆任务、8 个工具自动调用、权限确认、历史摘要、可中断 |
| 📊 数据分析 | 上传 CSV/Excel，自动统计、画直方图/相关性热力图、中文结论 |
| 📝 NLP | 情感判别、关键词提取、摘要、实体识别、文本分类 |
| 🗂️ 会话管理 | SQLite 持久化、多会话切换、一键导出 Markdown |

## 技术栈

- **后端**：Flask（工厂模式 + 蓝图分层）
- **大模型**：OpenAI 兼容接口（DeepSeek / 通义千问）、SSE 流式
- **Agent**：Function Calling、ReAct、Planner
- **数据**：pandas、numpy、matplotlib
- **NLP**：nltk、scikit-learn
- **存储**：SQLite
- **工程**：pytest、GitHub Actions CI、Docker

## 项目结构

```
smart-assistant/
├── app.py
├── config.py
├── routes/          # 路由层
│   ├── chat.py
│   ├── chart.py
│   ├── nlp.py
│   └── agent.py
├── services/        # 业务层
│   ├── llm_service.py
│   ├── chart_service.py
│   ├── nlp_service.py
│   ├── agent.py
│   └── db.py
├── utils/
├── templates/
├── tests/
└── .github/workflows/
```

## 快速运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
python run.py
```

访问 <http://127.0.0.1:5000/>
