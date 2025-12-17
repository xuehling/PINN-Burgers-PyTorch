import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- 配置 ---
NU = 0.01 / torch.pi  # 粘度系数

# --- 1. 定义神经网络 ---
# 输入变成 2 维 (x, t)，输出 1 维 (u)
# --- 修改 1: 自定义模型类，植入硬约束 ---
class HardConstraintPINN(nn.Module):
    def __init__(self):
        super(HardConstraintPINN, self).__init__()
        # 还是原来的网络结构，负责拟合剩余部分
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, inputs):
        # inputs 是 [N, 2]，其中第 0 列是 x，第 1 列是 t
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]

        # 先算出神经网络的原始输出
        net_out = self.net(inputs)

        # 【核心魔法】硬约束公式
        # 当 x = 1 或 x = -1 时，(1 - x^2) 必定为 0
        # 这就强行保证了边界输出为 0，无论 net_out 是多少
        u_final = (1.0 - x ** 2) * net_out

        return u_final


# 实例化模型
model = HardConstraintPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# --- 2. 数据准备 (Collocation Points) ---
# 我们在 x ∈ [-1, 1], t ∈ [0, 1] 的区域内随机撒点
# 相比于 meshgrid，随机撒点(Collocation)在高维问题中更常用
def get_pde_data(n=2000):
    # 随机生成 x 和 t
    x = torch.rand(n, 1) * 2 - 1  # [-1, 1]
    t = torch.rand(n, 1) * 1  # [0, 1]

    # 【关键】把 x 和 t 拼起来变成 [N, 2] 的矩阵
    # inputs[:, 0] 是 x, inputs[:, 1] 是 t
    inputs = torch.cat([x, t], dim=1).requires_grad_(True)
    return inputs


# --- 3. 训练循环 ---
print("开始训练 Burgers' Equation...")

for step in range(5001):  # 跑多一点，因为问题复杂

    # -------------------------
    # Part A: 计算 PDE Loss
    # -------------------------
    inputs = get_pde_data()
    u = model(inputs)

    # 第一次求导：得到 [u_x, u_t]
    # create_graph=True 必须开，因为后面还要对 u_x 求二阶导
    grads = torch.autograd.grad(u, inputs, torch.ones_like(u), create_graph=True)[0]

    # 【核心操作】切片！
    u_x = grads[:, 0:1]  # 取第0列，保持形状为 [N, 1]
    u_t = grads[:, 1:2]  # 取第1列，保持形状为 [N, 1]

    # 第二次求导：得到 u_xx
    # 注意：我们要对 u_x 求导，而不是 u
    grads_2 = torch.autograd.grad(u_x, inputs, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grads_2[:, 0:1]  # 只要 x 方向的二阶导

    # 组装 Burgers 方程残差: f = u_t + u*u_x - nu*u_xx
    f = u_t + u * u_x - NU * u_xx
    loss_pde = torch.mean(f ** 2)

    # -------------------------
    # Part B: 计算 初始条件 Loss (IC: t=0)
    # -------------------------
    # 生成 t=0 的数据点
    x_ic = torch.rand(500, 1) * 2 - 1  # x 随机
    t_ic = torch.zeros(500, 1)  # t = 0
    inputs_ic = torch.cat([x_ic, t_ic], dim=1)

    u_ic_pred = model(inputs_ic)
    u_ic_true = -torch.sin(torch.pi * x_ic)  # 初始波形
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    # -------------------------
    # Part C: 计算 边界条件 Loss (BC: x=-1 和 x=1)
    # -------------------------
    t_bc = torch.rand(500, 1)  # t 随机
    x_bc_left = -1 * torch.ones(500, 1)  # x = -1
    x_bc_right = 1 * torch.ones(500, 1)  # x = 1

    # 左右边界拼起来一起算
    inputs_bc = torch.cat([
        torch.cat([x_bc_left, t_bc], dim=1),
        torch.cat([x_bc_right, t_bc], dim=1)
    ], dim=0)

    u_bc_pred = model(inputs_bc)
    # 边界要求 u = 0
    # ... 前面的 Part A (PDE Loss) 和 Part B (IC Loss) 保持不变 ...
    # ... 注意：现在 model(inputs) 自动会经过刚才写的 (1-x^2) 逻辑 ...

    # -------------------------
    # Part C: 计算 边界条件 Loss (BC)  <-- 【直接删除整段！】
    # -------------------------
    # 硬约束下，边界天然满足 u=0，误差恒为 0，不需要训练了。
    # 把原来的 inputs_bc, u_bc_pred, loss_bc 全部删掉。

    # -------------------------
    # Part D: 总 Loss
    # -------------------------
    # loss = loss_pde + loss_ic + loss_bc  <-- 原来的
    loss = loss_pde + loss_ic  # <-- 【现在的】省掉了一项！

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印的时候也去掉 BC
    if step % 500 == 0:
        print(f"Step {step}: Loss={loss.item():.6f} (PDE={loss_pde.item():.5f}, IC={loss_ic.item():.5f})")

print("训练完成！准备绘图...")

# --- 4. 绘图 (画热力图) ---
# 生成网格数据用于画图
x_np = np.linspace(-1, 1, 100)
t_np = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_np, t_np)  # 生成网格

# 转换成 Tensor 喂给模型
x_flat = X.flatten()[:, None]
t_flat = T.flatten()[:, None]
inputs_test = torch.from_numpy(np.hstack((x_flat, t_flat))).float()

u_pred = model(inputs_test).detach().numpy()
U = u_pred.reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.pcolormesh(T, X, U, cmap='jet ', shading='auto')
plt.colorbar(label='u(x, t)')
plt.xlabel('t (Time)')
plt.ylabel('x (Space)')
plt.title("Burgers' Equation Solution (PINN)")
plt.show()