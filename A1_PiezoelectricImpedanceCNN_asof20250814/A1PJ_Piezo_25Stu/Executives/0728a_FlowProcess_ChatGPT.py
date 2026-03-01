from graphviz import Digraph

dot = Digraph(comment='Neural Network Architecture')

# Input layer
dot.node('A', 'Input Data (2496x1)')

# Initial Conv layer
dot.node('B', 'Conv1D: filters=128, kernel=7, activation=ReLU')

# MaxPooling layer
dot.node('C', 'MaxPooling1D: pool_size=2')

# Residual Block
dot.node('D', 'Residual Block\nConv1D, LayerNorm, Dropout')

# Attention Mechanisms
dot.node('E', 'Channel Attention')
dot.node('F', 'Temporal Attention')

# Global Average Pooling
dot.node('G', 'GlobalAveragePooling1D')

# Dense layer
dot.node('H', 'Dense: 64 units, activation=ReLU')

# Dropout
dot.node('I', 'Dropout: 0.3')

# Output
dot.node('J', 'Output: Dense(1)')

# Residual Block 2
dot.node('L', 'Residual Block\nConv1D, LayerNorm, Dropout')

# Edges connecting layers
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('D', 'L')
dot.edge('L', 'F')

# Highlight key components
dot.attr(style='dotted', color='blue')
dot.edge('D', 'E')
dot.edge('E', 'F')

# Render the graph to a file
dot.render('neural_network_architecture', format='png', view=True)
