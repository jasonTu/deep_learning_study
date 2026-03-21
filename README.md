# Deep Learning 分享实验说明

## 目录结构

```
/home/tuyoutu/sharing/deep_learning/
├── slides/
│   └── deep_learning_intro.pptx      # PPT文件
├── experiments/
│   ├── experiment_1_numpy_nn.py      # 实验一：手动实现神经网络
│   └── experiment_2_face_recognition.py  # 实验二：人脸识别
└── data/                              # 数据目录
    ├── face_dataset/                  # 人脸照片数据集
    ├── decision_boundary.png          # 决策边界可视化
    └── loss_curve.png                 # 损失曲线
```

## 实验一：手动实现神经网络

### 前置要求
```bash
pip install numpy matplotlib scikit-learn
```

### 运行
```bash
cd /home/tuyoutu/sharing/deep_learning
python experiments/experiment_1_numpy_nn.py
```

### 实验内容
1. **数据生成**: 使用半月形数据集（非线性可分）
2. **网络架构**: 2输入 → 5隐藏 → 1输出
3. **激活函数**: tanh(隐藏层) + sigmoid(输出层)
4. **训练过程**: 
   - 正向传播计算预测
   - 计算交叉熵损失
   - 反向传播计算梯度
   - 梯度下降更新参数
5. **可视化**: 决策边界、损失曲线

### 关键代码解释
```python
# 正向传播
Z1 = W1·X + b1      # 线性变换
A1 = tanh(Z1)       # 激活
Z2 = W2·A1 + b2
A2 = sigmoid(Z2)    # 输出

# 反向传播
dZ2 = A2 - Y        # 输出层误差
dW2 = dZ2·A1ᵀ / m   # W2梯度
dZ1 = W2ᵀ·dZ2 ⊙ g'(Z1)  # 反向传播误差
dW1 = dZ1·Xᵀ / m    # W1梯度

# 参数更新
W := W - α·dW       # 梯度下降
```

## 实验二：人脸识别

### 前置要求
```bash
pip install opencv-contrib-python numpy pillow
```

### 运行
```bash
cd /home/tuyoutu/sharing/deep_learning
python experiments/experiment_2_face_recognition.py
```

### 使用流程

#### 1. 收集数据 (选项1)
- 输入人员姓名
- 按 `c` 捕获照片（建议每人15-20张）
- 按 `q` 退出
- 照片保存在 `data/face_dataset/{姓名}/`

#### 2. 训练模型 (选项2)
- 自动进行数据增强（翻转、旋转、亮度调整）
- 使用 LBPH (局部二值模式直方图) 算法
- 模型保存在 `data/face_recognition_model.pkl`

#### 3. 实时识别 (选项3)
- 打开摄像头实时识别
- 显示人名和置信度
- 按 `s` 截图保存
- 按 `q` 退出

#### 4. 单张识别 (选项4)
- 输入图片路径进行识别
- 结果保存为 `{原文件名}_result.jpg`

### LBPH 算法简介
- **Local Binary Patterns**: 局部二值模式
- 对光照变化不敏感
- 适合小样本训练
- 计算效率高，适合实时应用

## PPT 演讲建议

### 时间分配（建议60分钟）
| 部分 | 时间 | 内容 |
|-----|-----|-----|
| 理论讲解 | 30分钟 | 神经网络基础、训练概念、传播算法 |
| 实验一 | 15分钟 | 代码演示 + 运行 |
| 实验二 | 10分钟 | 现场收集照片 + 训练 + 识别 |
| Q&A | 5分钟 | 答疑 |

### 演示技巧
1. **实验一**: 先展示结果，再逐步解释代码
2. **实验二**: 现场互动，让观众参与拍照
3. **可视化**: 利用决策边界图解释模型学到了什么

## 常见问题

### Q: 模型准确率不高？
- 增加训练样本数量
- 调整学习率
- 增加隐藏层神经元数
- 增加迭代次数

### Q: 人脸识别失败？
- 确保光线充足
- 正面面对摄像头
- 收集更多不同角度的照片
- 降低置信度阈值

### Q: 需要GPU吗？
- 实验一：不需要，CPU即可
- 实验二：不需要，LBPH是轻量级算法

## 扩展学习

### 改进方向
1. **实验一**: 
   - 增加更多隐藏层
   - 尝试不同激活函数
   - 实现动量优化/Adam
   
2. **实验二**:
   - 使用深度学习模型（FaceNet、DeepFace）
   - 增加活体检测
   - 多目标同时识别

### 推荐阅读
- 《深度学习》(Goodfellow)
- CS231n (斯坦福计算机视觉)
- Andrew Ng Deep Learning Specialization
