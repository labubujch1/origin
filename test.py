# 1. 准备训练数据
from dao import Value
from model import MLP

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]  # 目标输出

# 2. 初始化神经网络：3个输入，2个隐藏层（每层4个神经元），1个输出
model = MLP(3, [4, 4, 1])

# 3. 训练循环
learning_rate = 0.05
for epoch in range(20):
  # 前向传播：计算预测值
  ypred = [model(x) for x in xs]

  # 计算均方误差损失 (loss = sum((y_pred - y_gt)^2))
  loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

  # 梯度清零
  model.zero_grad()

  # 反向传播计算梯度
  loss.backward()

  # 梯度下降更新权重
  for p in model.parameters():
    p.data -= learning_rate * p.grad

  print(f"Epoch {epoch:2d} | Loss: {loss.data:.6f}")

print("\n训练后的预测结果：")
for x, ygt in zip(xs, ys):
  pred = model(x)
  print(f"输入: {x} | 真实值: {ygt} | 预测值: {pred.data:.4f}")