import math


class Value:
  """存储一个标量及其梯度，支持自动求导。"""

  def __init__(self, data, _children=(), _op="", label=""):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "只支持int/float幂次"
    out = Value(self.data**other, (self,), f"**{other}")

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

    def _backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out

  def backward(self):
    # 拓扑排序计算图中的所有节点
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    # 自动反向传播
    self.grad = 1.0
    for v in reversed(topo):
      v._backward()

  def __neg__(self):  # -self
    return self * -1

  def __radd__(self, other):  # other + self
    return self + other

  def __sub__(self, other):  # self - other
    return self + (-other)

  def __rsub__(self, other):  # other - self
    return other + (-self)

  def __rmul__(self, other):  # other * self
    return self * other

  def __truediv__(self, other):  # self / other
    return self * other**-1

  def __rtruediv__(self, other):  # other / self
    return other * self**-1

  def __repr__(self):
    return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"