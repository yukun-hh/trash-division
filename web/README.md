# Trash Division Web 前端

基于 Gradio 的垃圾分类识别 Web 应用，上传图片即可预测垃圾类别。

## 依赖

除项目根目录 `requirements.txt` 外，Web 前端额外依赖：

| 包 | 版本 | 说明 |
|---|---|---|
| `gradio` | `>=4.0,<5.0` | Web UI 框架 |
| `pydantic` | `>=2.5,<2.10` | gradio 4.x 兼容性约束（新版会报 `"const" in schema` 错误） |

> 安装：`pip install gradio>=4.0,<5.0 pydantic>=2.5,<2.10`

## 启动前准备

1. **确保 `best_model.pth` 存在**  
   在项目根目录（`trash-division/`）下放置训练好的模型权重。如没有，先运行：
   ```bash
   cd .. && python Train.py
   ```

2. **安装依赖**（如还未安装）：
   ```bash
   pip install -r ../requirements.txt
   ```

## 启动

在 `web/` 目录下运行：

```bash
python app.py
```

或者在项目根目录运行：

```bash
python web/app.py
```

启动后浏览器会自动打开 `http://127.0.0.1:7860`。

## 配置说明

可在 `app.py` 底部 `demo.launch()` 中调整以下参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `server_name` | `127.0.0.1` | 本机访问。如需局域网内其他设备访问，改为 `0.0.0.0` |
| `server_port` | `7860` | 端口号，冲突时可换 |
| `share` | `False` | 改为 `True` 可生成临时公网链接分享给同学 |
| `inbrowser` | `True` | 启动后自动打开浏览器 |

## 兼容性

| 项 | 说明 |
|---|---|
| Python | `>=3.9,<3.10`（Gradio 5.x 需 Python 3.10+） |
| PyTorch | `>=1.10` |
| 设备 | 自动选择 CUDA > Intel XPU > Apple MPS > CPU |
