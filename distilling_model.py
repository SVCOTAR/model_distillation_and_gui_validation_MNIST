import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from torchvision.datasets import MNIST

#==设置随机种子
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark=True

train_dataset = MNIST(root='mnist_data/', train=True,download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='mnist_data/', train=False,download=True, transform=transforms.ToTensor())

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,pin_memory=(device.type == 'cuda'),num_workers=0,)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False,pin_memory=(device.type == 'cuda'),num_workers=0,)

class TeacherModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(TeacherModel,self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,1200)
        self.fc2=nn.Linear(1200,1200)
        self.fc3=nn.Linear(1200,num_classes)
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=x.view(-1,784)
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc2(x)
        x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc3(x)
        return x

#---1.teacher大模型的训练和评估
model=TeacherModel()
model=model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs=3
for epoch in range(epochs):
    model.train()
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #==前向预测
        preds=model(data)
        loss=criterion(preds,targets)
        
        #==反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Teacher] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))

teacher_model=model        

class StudentModel(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(StudentModel,self).__init__()
        self.relu=nn.ReLU()
        self.fc1=nn.Linear(784,20)
        self.fc2=nn.Linear(20,20)
        self.fc3=nn.Linear(20,num_classes)
        self.dropout= nn.Dropout(p=0.5)
    def forward(self,x):
        x=x.view(-1,784)
        x=self.fc1(x)
        #x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc2(x)
        #x=self.dropout(x)
        x=self.relu(x)
        
        x=self.fc3(x)
        return x


#----2.student简单模型的训练和评估
model=StudentModel()
model=model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

epochs=3
for epoch in range(epochs):
    model.train()
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #==前向预测
        preds=model(data)
        loss=criterion(preds,targets)
        
        #==反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Student] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))
    
 
#---------------蒸馏teacher大模型

#准备预训练好的教师模型
teacher_model.eval()

#准备新的学生模型
model=StudentModel()
model=model.to(device)
model.train()

temp= 4


#hard_loss
hard_loss=nn.CrossEntropyLoss()
#hard_loss 权重
alpha= 0.3

#soft_loss 
soft_loss=nn.KLDivLoss(reduction="batchmean")
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


#--3. 蒸馏模型训练和评估
epochs=3
for epoch in range(epochs):
    
    #==训练集上训练模型权重
    for data ,targets in tqdm(train_loader):
        data =data.to(device)
        targets=targets.to(device)
        
        #--techer 模型
        with torch.no_grad():
            teacher_preds=teacher_model(data)
        
        #--学生模型
        student_preds = model(data)
        
        #--计算 hard_loss
        student_loss=hard_loss(student_preds,targets)
        
        #--计算蒸馏后的预测结果及soft_loss
        distillation_loss=soft_loss(F.log_softmax(student_preds/temp,dim=1),F.softmax(teacher_preds/temp,dim=1))* (temp ** 2)
        
        #将hard_loss和soft_loss加权求和
        loss=alpha*distillation_loss+(1-alpha)*student_loss
        
        #方向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #测试集上评估模型性能
    model.eval()
    num_correct=0
    num_samples=0

    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            y=y.to(device)

            preds=model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        acc=(num_correct/num_samples).item()

    model.train()
    print('[Distill] Epoch:{} \t Accuracy:{:.4f}'.format(epoch+1,acc))


# ================================================================
# 4. 保存与导出 student 模型 —— 让模型可以被独立加载和部署
# ----------------------------------------------------------------
# 产物目录: student_model_artifacts/
#   ├── pytorch_model.bin       PyTorch 原生权重（state_dict）
#   ├── model.safetensors       HF 推荐格式（更安全 + mmap 友好）
#   ├── model.onnx              通用推理格式，跨语言/框架可用
#   ├── config.json             模型超参与元数据
#   └── README.md               使用示例
# ================================================================
import json
from pathlib import Path

ARTIFACTS_DIR = Path("student_model_artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

model.eval()

# (1) config.json：超参 + 输入输出协议 + 标签映射
config = {
    "architecture": "StudentMLP",
    "framework": "pytorch",
    "torch_version": torch.__version__,
    "in_features": 784,
    "hidden_size": 20,
    "num_classes": 10,
    "input_shape": [1, 28, 28],
    "input_dtype": "float32",
    "preprocessing": {
        "color_mode": "L",
        "resize": [28, 28],
        "scale": "pixel / 255.0  (torchvision.transforms.ToTensor)",
        "layout": "NCHW",
    },
    "output": {"name": "logits", "shape": [10], "post": "softmax for probabilities"},
    "id2label": {str(i): str(i) for i in range(10)},
    "label2id": {str(i): i for i in range(10)},
    "distillation": {"teacher": "TeacherModel(784->1200->1200->10)", "T": temp, "alpha": alpha},
}
with open(ARTIFACTS_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"[export] config.json written")

# (2) PyTorch 原生权重
torch.save(model.state_dict(), ARTIFACTS_DIR / "pytorch_model.bin")
print(f"[export] pytorch_model.bin written")

# (3) safetensors（可选；HF 生态推荐）
try:
    from safetensors.torch import save_file
    state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, str(ARTIFACTS_DIR / "model.safetensors"))
    print(f"[export] model.safetensors written")
except ImportError:
    print("[export][warn] safetensors 未安装，跳过 .safetensors 导出 (pip install safetensors)")

# (4) ONNX：跨框架/跨语言部署的通用格式
dummy_input = torch.randn(1, 1, 28, 28, device=device)
onnx_path = ARTIFACTS_DIR / "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
    #dynamo=False,  # 走旧版 TorchScript 导出器，不依赖 onnxscript
)
print(f"[export] model.onnx written ({onnx_path.stat().st_size / 1024:.1f} KB)")

# (5) README.md
readme_text = """# MNIST Student Model (Distilled)

通过知识蒸馏从一个 1200x1200 MLP teacher 蒸馏出的 20x20 小型 student。

## Inputs
- shape: `[N, 1, 28, 28]` float32
- pixel range: `[0.0, 1.0]`（即 `torchvision.transforms.ToTensor()` 的输出）
- color: 灰度图（1 通道）

## Outputs
- `logits`：shape `[N, 10]`，未归一化分数
- 转概率：`probs = softmax(logits, dim=-1)`
- 预测类别：`pred = logits.argmax(-1)`

## 用法 1：PyTorch
```python
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

m = StudentModel()
m.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))
m.eval()
```

## 用法 2：ONNX Runtime（推荐，无需 PyTorch）
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("model.onnx")
img = Image.open("digit.png").convert("L").resize((28, 28))
arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(1, 1, 28, 28)
logits = sess.run(None, {"pixel_values": arr})[0][0]
print("predict:", int(logits.argmax()))
```

## Web UI
项目根目录提供 `app_gradio.py`，运行 `python app_gradio.py` 启动浏览器 UI，
支持上传图片或鼠标绘制数字直接识别。
"""
(ARTIFACTS_DIR / "README.md").write_text(readme_text, encoding="utf-8")
print(f"[export] README.md written")

print(f"\n[OK] all artifacts saved to: {ARTIFACTS_DIR.resolve()}")



# ============================================================
# 5. 校验：确认 ONNX 推理结果与 PyTorch 一致
# ============================================================
try:
    import numpy as _np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_np = dummy_input.detach().cpu().numpy()
    onnx_logits = sess.run(None, {"pixel_values": x_np})[0]
    with torch.no_grad():
        torch_logits = model(dummy_input).cpu().numpy()
    diff = _np.abs(onnx_logits - torch_logits).max()
    print(f"[check] max |onnx - pytorch| = {diff:.2e}  (应 < 1e-4)")
except ImportError:
    print("[check][warn] 未安装 onnxruntime，跳过对齐校验。pip install onnxruntime 即可")