import graphviz

def visualize_cartpole_pinn_detailed():
    dot = graphviz.Digraph(comment='CartpolePINN Model Architecture', format='svg')
    dot.attr(rankdir='LR', size='12,8')

    # Input
    dot.node('input', 'Input Sequence\n(x, x_dot, theta, theta_dot, action)', shape='box')

    # LSTM
    with dot.subgraph(name='cluster_lstm') as c:
        c.attr(label='LSTM (2 layers)')
        c.node('lstm1', 'LSTM Layer 1\n128 units', shape='box')
        c.node('lstm2', 'LSTM Layer 2\n128 units', shape='box')
        c.edge('lstm1', 'lstm2')

    # Main Network
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='Main Network')
        c.node('linear1', 'Linear (128 → 512)', shape='box')
        c.node('silu1', 'SiLU', shape='ellipse')
        c.node('ln1', 'LayerNorm', shape='diamond')
        c.node('dropout', 'Dropout (0.2)', shape='diamond')
        c.node('linear2', 'Linear (512 → 512)', shape='box')
        c.node('silu2', 'SiLU', shape='ellipse')
        c.node('ln2', 'LayerNorm', shape='diamond')
        c.node('linear3', 'Linear (512 → 256)', shape='box')
        c.node('silu3', 'SiLU', shape='ellipse')
        c.node('ln3', 'LayerNorm', shape='diamond')
        c.node('linear4', 'Linear (256 → 128)', shape='box')

        c.edge('linear1', 'silu1')
        c.edge('silu1', 'ln1')
        c.edge('ln1', 'dropout')
        c.edge('dropout', 'linear2')
        c.edge('linear2', 'silu2')
        c.edge('silu2', 'ln2')
        c.edge('ln2', 'linear3')
        c.edge('linear3', 'silu3')
        c.edge('silu3', 'ln3')
        c.edge('ln3', 'linear4')

    # Output
    dot.node('output', 'Output Layer', shape='box')

    # Force Output
    dot.node('force_output', 'Force Output', shape='ellipse')

    # Friction Output (optional)
    dot.node('friction_output', 'Friction Output\n(optional)', shape='ellipse', style='dashed')

    # Connections
    dot.edge('input', 'lstm1')
    dot.edge('lstm2', 'linear1')
    dot.edge('linear4', 'output')
    dot.edge('output', 'force_output')
    dot.edge('output', 'friction_output', style='dashed')

    # Residual connection
    dot.edge('lstm2', 'output', style='dashed', color='red', label='Residual')

    return dot

# Generate and save the visualization as SVG
dot = visualize_cartpole_pinn_detailed()
dot.render('media/cartpole_pinn_architecture_detailed', format='svg', cleanup=True)
print("Detailed visualization saved as 'cartpole_pinn_architecture_detailed.svg'")