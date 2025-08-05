# GitHub Pages 部署说明

## 🚀 快速部署到GitHub Pages

### 步骤1: 上传代码到GitHub

1. 在GitHub上创建新仓库（如：`scpi-search-tool`）
2. 将以下文件上传到仓库：
   ```
   ├── index.html           # 主页面
   ├── scpi_data.js         # SCPI数据文件（1.3MB）
   ├── _config.yml          # Jekyll配置
   ├── .github/workflows/pages.yml  # 自动部署配置
   └── README.md            # 项目说明
   ```

### 步骤2: 启用GitHub Pages

1. 进入仓库的 **Settings** 页面
2. 滚动到 **Pages** 部分
3. 在 **Source** 中选择 **GitHub Actions**
4. 保存设置

### 步骤3: 等待部署完成

- 推送代码后，GitHub Actions会自动构建和部署
- 在仓库的 **Actions** 标签页可以查看部署进度
- 部署完成后，网站将在 `https://[username].github.io/[repository-name]` 可用

### 步骤4: 访问网站

部署成功后，您可以通过以下地址访问：
```
https://[您的用户名].github.io/[仓库名称]
```

## 📁 文件说明

### 核心文件
- **`index.html`**: 主页面，包含完整的搜索界面和JavaScript搜索算法
- **`scpi_data.js`**: 转换后的SCPI指令数据，包含1713条指令的完整信息
- **`_config.yml`**: Jekyll配置文件，设置网站基本信息
- **`.github/workflows/pages.yml`**: 自动部署配置

### 可选文件
- **`README.md`**: 项目文档和使用说明
- **`scpi_search.py`**: Python版本的搜索工具（可选）
- **`requirements.txt`**: Python依赖（如果需要本地运行Python版本）

## 🔧 本地测试

在上传到GitHub之前，您可以在本地测试网页：

1. 直接双击打开 `index.html` 文件
2. 或使用简单的HTTP服务器：
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Node.js
   npx serve .
   ```
3. 在浏览器中访问 `http://localhost:8000`

## ⚙️ 自定义配置

### 修改网站标题和描述
编辑 `_config.yml` 文件：
```yaml
title: 您的网站标题
description: 您的网站描述
url: "https://[您的用户名].github.io"
```

### 修改搜索算法
所有搜索逻辑都在 `index.html` 的 `<script>` 部分，您可以：
- 调整权重系数
- 添加新的同义词
- 修改搜索结果数量

## 🛠️ 故障排除

### 常见问题

1. **网站无法访问**
   - 检查仓库是否为public
   - 确认GitHub Pages已启用
   - 等待几分钟让部署完成

2. **搜索功能不工作**
   - 检查浏览器控制台是否有错误
   - 确认 `scpi_data.js` 文件已正确上传

3. **样式显示异常**
   - 检查文件路径是否正确
   - 确认所有CSS都在 `index.html` 内部

### 调试技巧

1. 按F12打开浏览器开发者工具
2. 查看Console标签页是否有错误信息
3. 在Network标签页检查文件是否正确加载

## 📱 移动端适配

网页已经针对移动设备进行了优化：
- 响应式布局
- 触摸友好的界面
- 适配不同屏幕尺寸

## 🔄 更新数据

如果需要更新SCPI指令数据：
1. 修改Python脚本中的数据转换部分
2. 重新生成 `scpi_data.js` 文件
3. 提交更新到GitHub

## 📈 性能优化

当前配置已经过优化：
- 数据文件压缩（1.3MB）
- 客户端搜索（无需服务器）
- 缓存友好的设计

---

**注意**: 确保您有GitHub账户的必要权限来创建和管理仓库。部署过程通常需要几分钟时间。 