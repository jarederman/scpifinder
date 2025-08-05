# SCPI指令智能搜索工具

基于AI向量匹配技术的SCPI指令搜索工具，支持自然语言查询，使用多种相似度计算方法融合，确保搜索结果精确。

## 功能特点

- 🧠 **智能向量匹配**: 使用transformer模型生成语义向量
- 🔍 **多算法融合**: 融合4种相似度计算方法，提高匹配精度
- 🌐 **中英文支持**: 支持中文和英文自然语言查询
- 📊 **详细结果**: 显示指令、描述、原文和各种相似度分数
- ⚡ **高效搜索**: 基于预计算向量，搜索速度快

## 技术架构

### 相似度计算方法
- **余弦相似度** (权重: 40%) - 主要方法，计算向量夹角相似度
- **欧氏距离相似度** (权重: 30%) - 计算向量空间距离
- **曼哈顿距离相似度** (权重: 20%) - 计算L1范数距离
- **点积相似度** (权重: 10%) - 直接计算向量点积

### 向量生成
- 使用`sentence-transformers/all-MiniLM-L6-v2`模型
- 384维向量表示
- 支持中英文文本编码

## 安装依赖

```bash
# 进入conda环境
conda activate findCommand

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 交互模式（推荐）

```bash
python scpi_command_search.py
```

进入交互模式后，可以输入自然语言问题：
- `如何设置采集模式`
- `查看频率测量`
- `触发配置`
- `acquire measurement`
- `trigger settings`

### 2. 命令行模式

```bash
# 基本搜索
python scpi_command_search.py --query "acquire" --top_k 5

# 指定JSON文件
python scpi_command_search.py -f all_scpi_commands.json -q "measurement" -k 10

# 指定模型
python scpi_command_search.py -m "distilbert-base-uncased" -q "trigger"
```

### 3. 参数说明

- `--json_file, -f`: SCPI指令JSON文件路径 (默认: all_scpi_commands.json)
- `--model, -m`: transformer模型名称 (默认: sentence-transformers/all-MiniLM-L6-v2)
- `--query, -q`: 直接执行查询（非交互模式）
- `--top_k, -k`: 返回结果数量 (默认: 10)

## 搜索示例

### 示例1: 采集相关指令
```bash
$ python scpi_command_search.py -q "acquire" -k 3
```

结果：
- `ACQuire:SEQuence:CURrent?` - 返回序列中已完成到目前为止的采集次数
- `ACQuire:FASTAcq:PALEtte` - 设置或查询快速采集模式使用的调色板
- `ACQuire:STATE` - 启动或停止采集

### 示例2: 中文查询
```bash
$ python scpi_command_search.py -q "如何测量频率" -k 5
```

### 示例3: 交互模式
```
🔍 SCPI指令智能搜索工具
============================================================
请输入您的问题: 如何设置触发
正在搜索: '如何设置触发'
...
```

## 搜索技巧

1. **使用英文关键词**: SCPI指令本身是英文，使用英文关键词（如"acquire", "trigger", "measure"）通常能获得更精确的结果

2. **简洁明确**: 避免过于复杂的句子，使用关键术语

3. **多种表达方式**: 如果结果不理想，尝试不同的表达方式：
   - "acquire" vs "acquisition" vs "采集"
   - "trigger" vs "triggering" vs "触发"

4. **查看详细信息**: 每个结果都包含完整的原文描述，可以获取更多上下文信息

## 输出格式

每个搜索结果包含：
- **排名和综合得分**: 融合多种算法的最终分数
- **指令**: SCPI指令本身
- **描述**: 中文描述
- **原文**: 英文原始文档内容
- **相似度详情**: 各种算法的分数

## 常见问题

### Q: 为什么有些中文查询结果不准确？
A: 建议使用英文关键词，因为SCPI指令和原文都是英文，向量匹配对英文效果更好。

### Q: 如何提高搜索精度？
A: 
1. 使用具体的SCPI术语
2. 尝试英文关键词
3. 查看多个结果，综合判断

### Q: 模型加载失败怎么办？
A: 脚本会自动使用备用模型`distilbert-base-uncased`，或者检查网络连接。

## 文件说明

- `scpi_command_search.py`: 主程序
- `all_scpi_commands.json`: SCPI指令数据库（包含预计算向量）
- `requirements.txt`: Python依赖
- `README_SCPI_Search.md`: 使用说明

## 开发说明

### 权重调整
可以在`weighted_fusion`方法中调整各算法权重：

```python
weights = {
    'cosine': 0.4,      # 余弦相似度
    'euclidean': 0.3,   # 欧氏距离
    'manhattan': 0.2,   # 曼哈顿距离
    'dot_product': 0.1  # 点积
}
```

### 扩展功能
- 支持自定义权重参数
- 添加更多相似度算法
- 实现结果缓存
- 支持批量查询

---

**作者**: AI助手  
**版本**: 1.0  
**最后更新**: 2024年 