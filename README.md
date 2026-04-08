# 人智交互自适应支持系统

> **Adaptive Human-AI Interaction Support System**  
> Powered by DeepSeek & Streamlit

一个面向认知科学实验设计的智能对话系统。系统能够根据用户状态动态调整 AI 的回复风格，实现个性化的交互体验。

---

## 功能特性

- **自适应回复策略**：根据用户输入特征动态调整 AI 回复风格
- **思维导图识别**：支持上传思维导图图片，自动识别并转录
- **多任务支持**：内置三个不同主题的实验任务
- **实验日志记录**：自动记录每轮对话的完整数据

---

## 环境要求

- Python 3.10+
- macOS / Linux / Windows

---

## 安装

```bash
# 克隆项目
git clone <your-repo-url>
cd DialogSystem

# 安装依赖
pip install -r requirements.txt
```

---

## 配置

在项目根目录创建 `.streamlit/secrets.toml` 文件：

```toml
# DeepSeek API
DEEPSEEK_API_KEY = "your_deepseek_api_key"

# 百度 NLP API
APP_ID     = "your_baidu_app_id"
API_KEY    = "your_baidu_api_key"
SECRET_KEY = "your_baidu_secret_key"

# 百度千帆模型 API
BAIDU_VL_API_KEY = "qianfan_api_key"

# 邮件推送（可选，EMAIL_SENDER填NO或不填则不发送）
EMAIL_SENDER   = "sender@xx.com"        # 填 "NO" 跳过邮件发送，或填写实际邮箱
EMAIL_PASSWORD = "smtp_auth_code"
EMAIL_RECEIVER = "receiver@xx.com"
```

---

## 运行

```bash
streamlit run app.py
```

启动后按提示选择组别和被试编号即可开始实验。

---

## 项目结构

```
.
├── app.py              # 主程序：Streamlit 界面
├── strategy.py         # 策略决策模块
├── chat.py             # 对话生成模块
├── state_aware.py      # 状态感知模块
├── utils.py            # 工具函数
├── logger.py           # 日志配置
├── requirements.txt    # 依赖列表
└── read_log.ipynb      # 日志分析 Notebook
```

---

## 内置实验任务

| 任务 | 主题 |
|------|------|
| 任务一 | 低空经济：定义、特征、分类与发展措施 |
| 任务二 | 适老智能手机：类型、功能、价位评估 |
| 任务三 | 症状溯源：发热/恶心/腹泻/打鼾对应病症判断 |

每组被试需先围绕任务主题使用 AI 搜索并整理思维导图，再通过本系统进行深化追问。
