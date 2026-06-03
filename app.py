import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from Model import Net  # 根据上传的 Model.py，模型类名为 Net

# 1. 基础配置与类别映射
# 根据 Merge_classes.py，1=厨余垃圾, 2=可回收物, 3=其他垃圾, 4=有害垃圾
class_names = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']

# 设备自动选择逻辑，保持与 Train.py 和 Evaluate.py 一致
device = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print(f"当前使用的推理设备: {device}")

# 2. 初始化模型并加载最佳权重
model = Net(num_classes=4)
try:
    # 采用与 Evaluate.py 一致的健壮加载方式
    state_dict = torch.load('best_model.pth', map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
        
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("✅ 成功加载 best_model.pth 权重")
except Exception as e:
    print(f"⚠️ 模型加载失败，请确保目录下存在 best_model.pth。错误信息: {e}")

# 3. 定义数据预处理流程 (必须与 Evaluate.py 中的 val_transform 保持完全一致)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. 核心推理函数
def predict(image):
    if image is None:
        return None
    
    # Gradio 传入的 pil 图像，确保转为 RGB 格式
    image = image.convert('RGB')
    
    # 预处理并增加 batch 维度
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # 使用 Softmax 将 logits 转换为 0~1 的概率分布
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # 组装为 Gradio Label 组件需要的字典格式 { "类别名": 概率值 }
    result_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return result_dict

# 5. 构建与美化 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft(), title="Trash Division 垃圾分类识别") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>🗑️ Trash Division - 智能垃圾分类系统</h1>
            <p>基于 <b>ResNet-34</b> 架构，支持精准识别：<b>厨余垃圾、可回收物、其他垃圾、有害垃圾</b>。</p>
            <p><i>同济大学 Python 人工智能程序设计课程小组作业</i></p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # type="pil" 让 Gradio 直接传 PIL Image 对象给预测函数，配合 torchvision 最方便
            image_input = gr.Image(type="pil", label="上传垃圾图片 (支持拍照)")
            with gr.Row():
                clear_btn = gr.Button("清空", variant="secondary")
                submit_btn = gr.Button("开始识别", variant="primary")
        
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=4, label="预测结果与置信度")
            
    # 绑定点击事件
    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)
    clear_btn.click(lambda: (None, None), inputs=None, outputs=[image_input, label_output])

if __name__ == "__main__":
    # 启动 Web 界面
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 如果你想生成临时公网链接分享给同学测试，改为 True
        inbrowser=True  # 运行后自动在默认浏览器中打开
    )