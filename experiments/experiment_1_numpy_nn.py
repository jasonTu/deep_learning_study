"""
实验一：最简单的深度学习实践
使用 NumPy 手动实现神经网络，理解正向/反向传播
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 设置随机种子保证可重复
np.random.seed(42)

# ============================================
# 1. 生成数据集（半月形数据，二分类问题）
# ============================================
def load_data():
    """生成半月形数据集"""
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 可视化数据
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')
    plt.title('Training Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolors='k')
    plt.title('Test Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('/Users/tuyouwu/sharing/deep_learning/data/dataset_visualization.png')
    print("数据可视化已保存")
    
    return X_train.T, y_train.reshape(1, -1), X_test.T, y_test.reshape(1, -1)

# ============================================
# 2. 激活函数
# ============================================
def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0, z)

def relu_derivative(z):
    """ReLU 导数"""
    return (z > 0).astype(float)

def sigmoid_derivative(z):
    """Sigmoid 导数"""
    s = sigmoid(z)
    return s * (1 - s)

# ============================================
# 3. 网络初始化
# ============================================
def initialize_parameters(n_x, n_h, n_y):
    """
    初始化网络参数
    
    参数:
    n_x -- 输入层神经元数
    n_h -- 隐藏层神经元数
    n_y -- 输出层神经元数
    
    返回:
    parameters -- 包含W1, b1, W2, b2的字典
    """
    np.random.seed(42)
    
    W1 = np.random.randn(n_h, n_x) * 0.01  # 小随机数初始化
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters

# ============================================
# 4. 正向传播
# ============================================
def forward_propagation(X, parameters):
    """
    正向传播
    
    参数:
    X -- 输入数据 (n_x, m)
    parameters -- 网络参数
    
    返回:
    A2 -- 输出层激活值
    cache -- 缓存中间结果用于反向传播
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # 第一层
    Z1 = np.dot(W1, X) + b1      # 线性部分
    A1 = np.tanh(Z1)             # 激活函数（使用tanh）
    
    # 输出层
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)             # 二分类用sigmoid
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    
    return A2, cache

# ============================================
# 5. 计算损失
# ============================================
def compute_cost(A2, Y):
    """
    计算交叉熵损失
    
    参数:
    A2 -- 预测值 (1, m)
    Y -- 真实标签 (1, m)
    
    返回:
    cost -- 损失值
    """
    m = Y.shape[1]
    
    # 交叉熵损失
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m
    
    cost = float(np.squeeze(cost))  # 确保是标量
    
    return cost

# ============================================
# 6. 反向传播
# ============================================
def backward_propagation(parameters, cache, X, Y):
    """
    反向传播计算梯度
    
    参数:
    parameters -- 网络参数
    cache -- 正向传播的缓存
    X -- 输入数据
    Y -- 真实标签
    
    返回:
    grads -- 各参数的梯度
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # 输出层梯度
    dZ2 = A2 - Y                          # σ'(z) = a(1-a)，与损失结合简化
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # 隐藏层梯度
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # tanh导数: 1-a²
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    
    return grads

# ============================================
# 7. 参数更新
# ============================================
def update_parameters(parameters, grads, learning_rate=0.01):
    """
    使用梯度下降更新参数
    
    参数:
    parameters -- 当前参数
    grads -- 梯度
    learning_rate -- 学习率
    
    返回:
    parameters -- 更新后的参数
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # 更新
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters

# ============================================
# 8. 模型训练
# ============================================
def train_model(X, Y, n_h, num_iterations=10000, learning_rate=0.01, print_cost=True):
    """
    完整训练流程
    
    参数:
    X -- 训练数据
    Y -- 训练标签
    n_h -- 隐藏层神经元数
    num_iterations -- 迭代次数
    learning_rate -- 学习率
    print_cost -- 是否打印损失
    
    返回:
    parameters -- 训练好的参数
    costs -- 损失历史
    """
    np.random.seed(42)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # 初始化
    print(f"初始化网络: 输入层{n_x} -> 隐藏层{n_h} -> 输出层{n_y}")
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    costs = []
    
    # 训练循环
    for i in range(num_iterations):
        # 正向传播
        A2, cache = forward_propagation(X, parameters)
        
        # 计算损失
        cost = compute_cost(A2, Y)
        
        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        
        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 记录损失
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
    
    return parameters, costs

# ============================================
# 9. 预测
# ============================================
def predict(parameters, X):
    """
    使用训练好的模型预测
    
    参数:
    parameters -- 训练好的参数
    X -- 输入数据
    
    返回:
    predictions -- 预测结果 (0 或 1)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int)
    return predictions

def predict_proba(parameters, X):
    """预测概率"""
    A2, cache = forward_propagation(X, parameters)
    return A2

# ============================================
# 10. 可视化决策边界
# ============================================
def plot_decision_boundary(parameters, X, y, title="Decision Boundary"):
    """绘制决策边界"""
    X = X.T  # 转置回 (m, 2)
    y = y.flatten()
    
    # 设置边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = predict(parameters, grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('/Users/tuyouwu/sharing/deep_learning/data/decision_boundary.png')
    print(f"决策边界图已保存")

# ============================================
# 主程序
# ============================================
def main():
    print("=" * 60)
    print("实验一：手动实现神经网络")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"训练集: {X_train.shape[1]} 个样本")
    print(f"测试集: {X_test.shape[1]} 个样本")
    print(f"特征维度: {X_train.shape[0]}")
    
    # 2. 训练模型
    print("\n[2/5] 训练模型...")
    n_h = 5  # 隐藏层5个神经元
    parameters, costs = train_model(
        X_train, Y_train, 
        n_h=n_h, 
        num_iterations=10000, 
        learning_rate=0.5,
        print_cost=True
    )
    
    # 3. 绘制损失曲线
    print("\n[3/5] 绘制训练曲线...")
    plt.figure(figsize=(8, 5))
    plt.plot(costs)
    plt.title('Training Loss')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.savefig('/Users/tuyouwu/sharing/deep_learning/data/loss_curve.png')
    print("损失曲线已保存")
    
    # 4. 评估模型
    print("\n[4/5] 评估模型...")
    train_predictions = predict(parameters, X_train)
    test_predictions = predict(parameters, X_test)
    
    train_accuracy = np.mean(train_predictions == Y_train) * 100
    test_accuracy = np.mean(test_predictions == Y_test) * 100
    
    print(f"训练集准确率: {train_accuracy:.2f}%")
    print(f"测试集准确率: {test_accuracy:.2f}%")
    
    # 5. 可视化
    print("\n[5/5] 可视化决策边界...")
    plot_decision_boundary(parameters, X_train, Y_train, "Training Set Decision Boundary")
    plot_decision_boundary(parameters, X_test, Y_test, "Test Set Decision Boundary")
    
    # 6. 预测示例
    print("\n[示例预测]")
    sample_idx = 0
    sample_x = X_test[:, sample_idx:sample_idx+1]
    sample_y = Y_test[:, sample_idx:sample_idx+1]
    
    prob = predict_proba(parameters, sample_x)
    pred = predict(parameters, sample_x)
    
    print(f"输入特征: [{sample_x[0,0]:.3f}, {sample_x[1,0]:.3f}]")
    print(f"真实标签: {sample_y[0,0]}")
    print(f"预测概率: {prob[0,0]:.4f}")
    print(f"预测结果: {pred[0,0]}")
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print(f"模型文件保存在当前目录")
    print("=" * 60)
    
    return parameters

if __name__ == "__main__":
    parameters = main()
