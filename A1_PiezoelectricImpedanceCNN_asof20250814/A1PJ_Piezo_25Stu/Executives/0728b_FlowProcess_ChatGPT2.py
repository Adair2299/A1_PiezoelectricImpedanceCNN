import matplotlib.pyplot as plt
import networkx as nx

# 创建一个图
G = nx.DiGraph()

# 添加节点（神经网络层）
G.add_node("Input\n(2496x1)")
G.add_node("Conv1D\n(filters=128, kernel=7)")
G.add_node("MaxPooling1D\n(pool_size=2)")
G.add_node("Residual Block\n(Conv1D, LayerNorm, Dropout)")
G.add_node("Channel Attention")
G.add_node("Temporal Attention")
G.add_node("GlobalAveragePooling1D")
G.add_node("Dense\n(64 units, activation=ReLU)")
G.add_node("Dropout\n(0.3)")
G.add_node("Output\n(Dense 1)")

# 添加边（表示层与层之间的连接）
G.add_edges_from([
    ("Input\n(2496x1)", "Conv1D\n(filters=128, kernel=7)"),
    ("Conv1D\n(filters=128, kernel=7)", "MaxPooling1D\n(pool_size=2)"),
    ("MaxPooling1D\n(pool_size=2)", "Residual Block\n(Conv1D, LayerNorm, Dropout)"),
    ("Residual Block\n(Conv1D, LayerNorm, Dropout)", "Channel Attention"),
    ("Channel Attention", "Temporal Attention"),
    ("Temporal Attention", "GlobalAveragePooling1D"),
    ("GlobalAveragePooling1D", "Dense\n(64 units, activation=ReLU)"),
    ("Dense\n(64 units, activation=ReLU)", "Dropout\n(0.3)"),
    ("Dropout\n(0.3)", "Output\n(Dense 1)")
])

# 绘制图形
pos = nx.spring_layout(G, seed=42)  # 使用Spring布局来排布节点
plt.figure(figsize=(10, 10))  # 设置图形大小
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray', width=2)
plt.title("Neural Network Architecture")
plt.show()
