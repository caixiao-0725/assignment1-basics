#!/usr/bin/env python3
"""
RoPE 小例子：2 个 token × 6 维向量，逐步打印中间量，对照 tests/adapters.py 里的逻辑。

运行：uv run python scripts/rope_explained.py
"""

from __future__ import annotations

import torch

# --- 与作业一致的形态约定 -------------------------------------------------
# Q 形状: (sequence_length, d_k) = (2, 6)
# 即：第 0 行是位置 0 的 query 向量，第 1 行是位置 1 的 query 向量。
# adapters 里还有更前面的 batch、head 维，这里省略，只看最后两维。

torch.manual_seed(42)
Q = torch.randn(2, 6)
d_k = 6
theta = 10_000.0
# 两个 token 的位置编号：0 和 1（和你作业里 torch.arange(seq) 一样）
token_positions = torch.tensor([0, 1], dtype=torch.float)

print("=" * 60)
print("1) 输入 Q，形状 (seq_len=2, d_k=6)：每行是一个位置的 6 维向量")
print("=" * 60)
print(Q)
print()

# --- inv_freq：每个「维度对」的基础角速度 -----------------------------------
# 维度按 (0,1), (2,3), (4,5) 两两配对；共 d_k//2 = 3 对。
# 第 j 对（j=0,1,2）满足： inv_freq[j] = theta^(-2j/d_k) = 1 / theta^(2j/d_k)
# torch.arange(0, d_k, 2) = [0, 2, 4]，除以 d_k 得到 [0, 2/6, 4/6]
exp = torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
inv_freq = 1.0 / (theta**exp)

print("=" * 60)
print("2) 只对「成对的下标」算频率：arange(0, d_k, 2) -> 指数里用 2j/d_k")
print("   exp =", exp.tolist())
print("   inv_freq (长度 3，对应 3 对维度):", inv_freq.tolist())
print("   含义：位置 m 上，第 j 对的旋转角 = m * inv_freq[j]（弧度）")
print()

# --- 每个位置的角度：angles[pos, j] = pos_id[pos] * inv_freq[j] ------------
# 这里 leading 为空，相当于 x 形状 (2,6), pos 先变成 (2,1) 再乘 inv_freq (3,)
seq_len = Q.shape[0]
pos = token_positions.reshape(seq_len, 1)  # (2, 1)
angles = pos * inv_freq  # 广播 -> (2, 3)

print("=" * 60)
print("3) token_positions 变成列向量 (2,1)，乘 inv_freq (3,) 得到 angles (2,3)")
print("   token_positions:", token_positions.tolist())
print("   angles[位置, 第几对]:\n", angles)
print()

cos = angles.cos()
sin = angles.sin()

print("=" * 60)
print("4) cos/sin 与 angles 同形状 (2, 3)；对每一行、每一对各有一对 cos/sin")
print("cos:\n", cos)
print("sin:\n", sin)
print()

# --- 分拆偶数位 / 奇数位，做 2D 旋转 -----------------------------------------
# x1 = Q[..., 0::2] 取下标 0,2,4；x2 取下标 1,3,5
x1 = Q[..., 0::2]
x2 = Q[..., 1::2]
print("=" * 60)
print("5) 把最后维拆成两截：x1 = Q[..., 0,2,4]，x2 = Q[..., 1,3,5]")
print("x1:\n", x1)
print("x2:\n", x2)
print()

y1 = x1 * cos - x2 * sin
y2 = x1 * sin + x2 * cos

print("=" * 60)
print("6) 逐对旋转（对每个位置、每一对 j 同时算）：")
print("   y1[:,j] = x1[:,j]*cos[:,j] - x2[:,j]*sin[:,j]")
print("   y2[:,j] = x1[:,j]*sin[:,j] + x2[:,j]*cos[:,j]")
print("y1:\n", y1)
print("y2:\n", y2)
print()

Q_rope = torch.empty_like(Q)
Q_rope[..., 0::2] = y1
Q_rope[..., 1::2] = y2

print("=" * 60)
print("7) 交错拼回 6 维：下标 0,1,2,3,4,5 <- (y1 的 0,1,2) 与 (y2 的 0,1,2)")
print("RoPE 后的 Q_rope:")
print(Q_rope)
print()

# --- 手算一组，验证「就是一个 2x2 旋转矩阵」 ------------------------------
print("=" * 60)
print("8) 手算核对：位置 1 上第 0 对 (原下标 0 和 1)")
pos_val = 1
j = 0
angle = float(pos_val * inv_freq[j])
c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
q0, q1 = float(Q[1, 0]), float(Q[1, 1])
yy0 = c * q0 - s * q1
yy1 = s * q0 + c * q1
print(f"   位置 pos={pos_val}, 第 {j} 对, angle={angle:.6f} rad")
print(f"   [q0,q1]=[{q0:.4f},{q1:.4f}] -> [{yy0:.4f},{yy1:.4f}]")
print(f"   与上面 y1[1,0],y2[1,0] 应一致: {float(y1[1,0]):.4f}, {float(y2[1,0]):.4f}")
print("=" * 60)

print("\n对照 adapters.run_rope：把 Q 看成 in_query_or_key，最前可再加 batch/head 维；")
print("broadcast_to 用来让 token_positions 和 (*leading, seq) 形状对齐。")
