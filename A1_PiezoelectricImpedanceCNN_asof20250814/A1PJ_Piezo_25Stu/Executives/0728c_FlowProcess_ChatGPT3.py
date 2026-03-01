import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow, FancyArrowPatch
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

# 1. 设置中文字体（关键修复）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


sequence_length = 2496

plt.figure(figsize=(20, 25))
ax = plt.gca()
ax.set_xlim(0, 20)
ax.set_ylim(0, 25)
ax.invert_yaxis()
ax.axis('off')

# 颜色定义
colors = {
    'input': '#FFDDC1',
    'conv': '#C1FFD7',
    'pool': '#C1E1FF',
    'norm': '#FFC1E1',
    'dropout': '#E1C1FF',
    'attention': '#FFFAC1',
    'dense': '#C1FFFF',
    'output': '#FFC1C1',
    'residual': '#D0D0D0',
    'connector': '#808080'
}


# 绘制区块函数
def draw_block(x, y, width, height, label, color, text_size=10, details=None):
    rect = Rectangle((x, y), width, height,
                     facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(rect)

    # 主标签
    plt.text(x + width / 2, y + height / 4, label,
             ha='center', va='center', fontsize=text_size + 2, weight='bold')

    # 详细参数
    if details:
        plt.text(x + width / 2, y + height * 0.65, details,
                 ha='center', va='center', fontsize=text_size)

    return (x + width / 2, y + height)


# 绘制连接线
def draw_connection(start, end, style='normal', label=None):
    if style == 'residual':
        # 残差连接
        arrow = FancyArrowPatch(start, (start[0], start[1] + 1.5),
                                arrowstyle='-', color=colors['connector'], lw=1.5)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((start[0], start[1] + 1.5), (end[0], end[1] + 1.5),
                                arrowstyle='-', color=colors['connector'], lw=1.5)
        ax.add_patch(arrow)
        arrow = FancyArrowPatch((end[0], end[1] + 1.5), end,
                                arrowstyle='->', color=colors['connector'], lw=1.5)
        ax.add_patch(arrow)
        plt.text((start[0] + end[0]) / 2, start[1] + 1.8, "跳跃连接",
                 ha='center', va='bottom', fontsize=9, color='darkred')
    else:
        # 普通连接
        arrow = FancyArrowPatch(start, end,
                                arrowstyle='->', color=colors['connector'], lw=1.5)
        ax.add_patch(arrow)
        if label:
            plt.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, label,
                     ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))


# 绘制标题
plt.text(10, 0.5, "高级时序神经网络架构",
         ha='center', va='center', fontsize=24, weight='bold')
plt.text(10, 1.2, "带双重注意力机制的残差卷积网络",
         ha='center', va='center', fontsize=18, color='darkblue')

# ===================== 输入部分 =====================
start_y = 2.5
input_end = draw_block(8, start_y, 4, 1, "输入层", colors['input'],
                       details=f"形状: ({sequence_length}, 1)\n通道: 1")

# ===================== 卷积部分 =====================
conv_end = draw_block(8, start_y + 1.5, 4, 1.5, "初始卷积层", colors['conv'],
                      details="Conv1D: 128滤波器\n核大小: 7\n激活: ReLU\n填充: same")
draw_connection(input_end, conv_end[:2])

pool_end = draw_block(8, start_y + 3.5, 4, 1, "最大池化层", colors['pool'],
                      details="池大小: 2\n输出尺寸: (seq_len/2, 128)")
draw_connection(conv_end[:2], pool_end[:2])

# ===================== 残差块部分 =====================
res_start = (pool_end[0], pool_end[1])
res_block_y = start_y + 5.0

# 残差块主路径
res_conv1 = draw_block(7.5, res_block_y, 2, 1, "卷积层", colors['conv'],
                       details="Conv1D: 128滤波器\n核大小: 3\n激活: ReLU\n填充: same")
draw_connection(res_start, (res_conv1[0], res_block_y - 0.2))

res_norm = draw_block(7.5, res_block_y + 1.5, 2, 0.7, "层归一化", colors['norm'])
draw_connection(res_conv1[:2], res_norm[:2])

res_drop = draw_block(7.5, res_block_y + 2.3, 2, 0.7, "Dropout层", colors['dropout'],
                      details="比率: 0.2")
draw_connection(res_norm[:2], res_drop[:2])

res_conv2 = draw_block(7.5, res_block_y + 3.2, 2, 1, "卷积层", colors['conv'],
                       details="Conv1D: 128滤波器\n核大小: 3\n填充: same")
draw_connection(res_drop[:2], res_conv2[:2])

# 残差连接
res_add = draw_block(9.5, res_block_y + 4.5, 2, 0.7, "相加操作", colors['residual'])
draw_connection(res_conv2[:2], (res_add[0], res_block_y + 4.5))
draw_connection(res_start, (9.5, res_block_y + 4.5), style='residual')

res_norm2 = draw_block(9.5, res_block_y + 5.4, 2, 0.7, "层归一化", colors['norm'])
draw_connection(res_add[:2], res_norm2[:2])

res_pool = draw_block(9.5, res_block_y + 6.3, 2, 0.7, "最大池化", colors['pool'],
                      details="池大小: 2\n输出尺寸: (seq_len/4, 128)")
draw_connection(res_norm2[:2], res_pool[:2])

# ===================== 注意力机制部分 =====================
attn_y = res_block_y + 7.5

# 通道注意力
ca_block = draw_block(5, attn_y, 4, 3.5, "通道注意力模块", colors['attention'])
ca_details = [
    "1. 全局平均池化 [B, T, C] → [B, 1, C]",
    "2. Dense(C/8, 激活=ReLU) 压缩率=8",
    "3. Dense(C, 激活=sigmoid) 生成通道权重",
    "4. 输入与权重逐元素相乘"
]
plt.text(ca_block[0], attn_y + 0.5, "\n".join(ca_details),
         ha='center', va='top', fontsize=10)
draw_connection(res_pool[:2], (5, attn_y))

# 时序注意力
ta_block = draw_block(11, attn_y, 4, 3.5, "时序注意力模块", colors['attention'])
ta_details = [
    "1. 维度置换 [B, T, C] → [B, C, T]",
    "2. Dense(T, 激活=softmax) 生成时序权重",
    "3. 维度还原 [B, C, T] → [B, T, C]",
    "4. 输入与权重逐元素相乘"
]
plt.text(ta_block[0], attn_y + 0.5, "\n".join(ta_details),
         ha='center', va='top', fontsize=10)
draw_connection((5, attn_y + 3.5), (11, attn_y))

# ===================== 解码器部分 =====================
decoder_y = attn_y + 4.5
gap_end = draw_block(8, decoder_y, 4, 1, "全局平均池化", colors['pool'],
                     details="输出尺寸: (128,)")
draw_connection((ca_block[0], attn_y + 3.5), (8, decoder_y))
draw_connection((ta_block[0], attn_y + 3.5), (8, decoder_y))

dense_end = draw_block(8, decoder_y + 1.5, 4, 1, "全连接层", colors['dense'],
                       details="单元: 64\n激活: ReLU")
draw_connection(gap_end[:2], dense_end[:2])

drop_end = draw_block(8, decoder_y + 3, 4, 0.8, "Dropout层", colors['dropout'],
                      details="比率: 0.3")
draw_connection(dense_end[:2], drop_end[:2])

output_end = draw_block(8, decoder_y + 4, 4, 1, "输出层", colors['output'],
                        details="Dense(1)\n线性激活 (回归输出)")
draw_connection(drop_end[:2], output_end[:2])

# ===================== 训练配置部分 =====================
config_y = decoder_y + 5.5
config_box = Rectangle((2, config_y), 16, 3.5, facecolor='#F0F8FF', edgecolor='navy', alpha=0.7)
ax.add_patch(config_box)

config_title = plt.text(10, config_y + 0.5, "模型训练配置",
                        ha='center', va='center', fontsize=16, weight='bold', color='navy')

config_details = [
    "优化器: Adam(learning_rate=0.001)",
    "损失函数: 均方误差(MSE)",
    "监控指标: 平均绝对误差(MAE)",
    "回调函数:",
    "  - ReduceLROnPlateau(监控='val_loss', 衰减系数=0.5, 耐心=3)",
    "  - EarlyStopping(监控='val_loss', 耐心=5, 恢复最佳权重)"
]

plt.text(10, config_y + 2, "\n".join(config_details),
         ha='center', va='center', fontsize=12)

# ===================== 设计亮点部分 =====================
features_y = config_y + 4.5
features_box = Rectangle((1, features_y), 18, 4, facecolor='#FFF0F5', edgecolor='purple', alpha=0.7)
ax.add_patch(features_box)

features_title = plt.text(10, features_y + 0.5, "架构设计亮点",
                          ha='center', va='center', fontsize=16, weight='bold', color='purple')

features = [
    "• 双重注意力机制: 通道注意力 + 时序注意力，聚焦关键特征",
    "• 残差连接: 解决梯度消失，稳定深层网络训练",
    "• 高效特征压缩: 4级时序压缩(初始池化+残差池化)",
    "• 强正则化组合: Dropout(0.2-0.3) + 全局池化 + 早停机制",
    "• 参数优化: 层归一化加速收敛，自适应学习率调整"
]

plt.text(10, features_y + 2.2, "\n".join(features),
         ha='center', va='center', fontsize=12)

# 保存图像
plt.tight_layout()
plt.savefig('advanced_neural_network_architecture.png', dpi=300, bbox_inches='tight')
plt.show()