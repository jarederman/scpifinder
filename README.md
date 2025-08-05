# SCPI指令智能搜索工具

基于AI向量匹配的SCPI指令搜索系统，支持Web界面和命令行两种使用方式。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确保在findCommand conda环境中
conda activate findCommand

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 启动Web应用 (推荐)

```bash
python app.py
```

访问 http://localhost:5001 打开Web界面

### 3. 命令行使用

```bash
# 交互模式
python scpi_search.py

# 直接查询
python scpi_search.py --query "acquire" --top_k 5

# 查看详细对比
python scpi_search.py --query "trigger" --compare
```

## 🌟 特性

- **🧠 智能向量匹配**: 使用transformer模型进行语义搜索
- **🎯 高精度结果**: 6种相似度算法融合，自适应权重优化
- **🌐 Web界面**: 现代化响应式设计，无需滚动查看所有结果
- **📱 移动友好**: 支持桌面和移动设备
- **⚡ 高性能**: 基于预计算向量，搜索速度快

## 💡 使用技巧

- **英文关键词效果更好**: "acquire", "trigger", "measurement"
- **具体描述优于模糊描述**: "frequency measurement" vs "measurement"
- **支持中英文查询**: 支持中文但英文精度更高
- **查看完整原文**: 点击"展开/收起"查看SCPI指令的完整文档

## 📊 搜索示例

| 查询 | 推荐结果 |
|------|---------|
| `acquire` | ACQuire:SEQuence:CURrent?, ACQuire? |
| `trigger` | TRIGger:A, TRIGger:B |
| `measurement frequency` | POWer:QUALity:FREQuency?, DVM:MEASUrement:FREQuency? |
| `频率测量` | 频率相关的SCPI指令 |

## 📁 文件说明

- `app.py` - Flask Web应用
- `scpi_search.py` - 命令行搜索工具
- `scpi_data.json` - SCPI指令数据库 (1713条指令)
- `requirements.txt` - Python依赖
- `templates/index.html` - Web界面模板
- `static/style.css` - Web界面样式

## 🔧 技术架构

- **前端**: HTML5 + CSS3 + Vanilla JavaScript
- **后端**: Flask + Python
- **AI模型**: sentence-transformers/all-MiniLM-L6-v2
- **向量计算**: 384维归一化向量 + 6种相似度算法
- **数据**: 1713条SCPI指令，预计算向量

---

**使用愉快！** 🎉 