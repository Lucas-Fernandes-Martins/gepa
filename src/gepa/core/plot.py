from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np




def plot_gepa_evolution_tree(optimized_results_dict: Dict[str, Any], 
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: str = None,
                           title: str = "GEPA Evolution Tree") -> plt.Figure:
    """
    Plot GEPA evolution tree
    
    Args:
        optimized_results_dict: Dictionary from optimized_results.to_dict()
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the plot
        title: Title for the plot
        
    Returns:
        matplotlib Figure object
    """
    
    # Extract data from GEPA results
    candidates = optimized_results_dict['candidates']
    parents = optimized_results_dict['parents']
    scores = optimized_results_dict['val_aggregate_scores']
    best_idx = optimized_results_dict['best_idx']
    
    # Build node data structure
    nodes = []
    for i, (candidate, parent_list, score) in enumerate(zip(candidates, parents, scores)):
        prompt = candidate.get('instruction_prompt', '')
        parent_idx = parent_list[0] if parent_list and parent_list[0] is not None else None
        
        node = {
            'id': i,
            'score': score,
            'prompt': prompt,
            'prompt_length_chars': len(prompt),
            'prompt_length_words': len(prompt.split()),
            'parent_idx': parent_idx,
            'is_best': i == best_idx,
            'level': 0  # Will be calculated
        }
        nodes.append(node)
    
    # Calculate levels (depth in tree)
    def calculate_level(node_id, nodes, memo={}):
        if node_id in memo:
            return memo[node_id]
        
        node = nodes[node_id]
        if node['parent_idx'] is None:
            level = 0
        else:
            level = calculate_level(node['parent_idx'], nodes, memo) + 1
        
        memo[node_id] = level
        node['level'] = level
        return level
    
    for node in nodes:
        calculate_level(node['id'], nodes)
    
    # Group nodes by level
    levels = {}
    for node in nodes:
        level = node['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(node['id'])
    
    # Build edges
    edges = []
    for node in nodes:
        if node['parent_idx'] is not None:
            parent_score = nodes[node['parent_idx']]['score']
            edge = {
                'from': node['parent_idx'],
                'to': node['id'],
                'score_delta': node['score'] - parent_score
            }
            edges.append(edge)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # Layout parameters
    level_height = 3.0
    node_width = 2.0
    node_height = 1.2
    margin = 1.0
    
    max_level = max(levels.keys())
    max_nodes_per_level = max(len(level_nodes) for level_nodes in levels.values())
    
    # Calculate positions for each node
    node_positions = {}
    for level, node_ids in levels.items():
        y = max_level * level_height - level * level_height  # Top to bottom
        
        # Center nodes horizontally
        total_width = len(node_ids) * node_width + (len(node_ids) - 1) * margin
        start_x = -total_width / 2
        
        for i, node_id in enumerate(node_ids):
            x = start_x + i * (node_width + margin) + node_width / 2
            node_positions[node_id] = (x, y)
    
    # Draw edges first (so they appear behind nodes)
    for edge in edges:
        from_pos = node_positions[edge['from']]
        to_pos = node_positions[edge['to']]
        
        # Draw line
        color = '#4caf50' if edge['score_delta'] >= 0 else '#f44336'
        linewidth = min(abs(edge['score_delta']) * 20 + 1, 8)  # Scale line width with score change
        alpha = 0.7
        
        ax.plot([from_pos[0], to_pos[0]], 
                [from_pos[1] - node_height/2, to_pos[1] + node_height/2], 
                color=color, linewidth=linewidth, alpha=alpha, zorder=1)
        
        # Add score delta label
        mid_x = (from_pos[0] + to_pos[0]) / 2
        mid_y = (from_pos[1] - node_height/2 + to_pos[1] + node_height/2) / 2
        
        delta_text = f"+{edge['score_delta']:.3f}" if edge['score_delta'] >= 0 else f"{edge['score_delta']:.3f}"
        ax.text(mid_x, mid_y, delta_text, 
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=3)
    
    # Draw nodes
    for node in nodes:
        pos = node_positions[node['id']]
        x, y = pos
        
        # Node styling
        if node['is_best']:
            facecolor = '#fff3e0'
            edgecolor = '#ff9800'
            linewidth = 3
        else:
            facecolor = '#f5f5f5'
            edgecolor = '#ddd'
            linewidth = 1
        
        # Draw node rectangle
        rect = FancyBboxPatch(
            (x - node_width/2, y - node_height/2),
            node_width, node_height,
            boxstyle="round,pad=0.1",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add best marker
        if node['is_best']:
            ax.text(x + node_width/2 - 0.2, y + node_height/2 - 0.2, 'B', 
                    ha='center', va='center', fontsize=12, zorder=4)
        
        # Node ID
        ax.text(x, y + 0.2, f"C{node['id']}", 
                ha='center', va='center', fontweight='bold', fontsize=11, zorder=4)
        
        # Score
        ax.text(x, y, f"{node['score']:.3f}", 
                ha='center', va='center', fontsize=9, color='#666', zorder=4)
        
        # Character count
        ax.text(x, y - 0.2, f"{node['prompt_length_chars']}ch", 
                ha='center', va='center', fontsize=8, color='#999', zorder=4)
    
    # Set plot limits and remove axes
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    
    x_margin = node_width
    y_margin = node_height
    
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title and metadata
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend/info
    info_text = f"Candidates: {len(nodes)} | Levels: {len(levels)} | Best: C{best_idx} (Score: {scores[best_idx]:.3f})"
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10, color='#666')
    
    # Add color legend
    legend_elements = [
        plt.Line2D([0], [0], color='#4caf50', linewidth=3, label='Score improvement'),
        plt.Line2D([0], [0], color='#f44336', linewidth=3, label='Score decline'),
        patches.Patch(facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2, label='Best candidate')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evolution tree saved to: {save_path}")
    
    plt.show()

    return fig

def print_evolution_summary(optimized_results_dict: Dict[str, Any]):
    """Print a text summary of the evolution tree."""
    
    candidates = optimized_results_dict['candidates']
    parents = optimized_results_dict['parents']
    scores = optimized_results_dict['val_aggregate_scores']
    best_idx = optimized_results_dict['best_idx']
    
    print("GEPA Evolution Tree Summary")
    print("=" * 50)
    print(f"Total candidates: {len(candidates)}")
    print(f"Best candidate: C{best_idx} (Score: {scores[best_idx]:.4f})")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    
    print(f"\n📊 Evolution path:")
    for i, (parent_list, score) in enumerate(zip(parents, scores)):
        parent_idx = parent_list[0] if parent_list and parent_list[0] is not None else None
        
        if parent_idx is None:
            print(f"  C{i}: Root node (Score: {score:.4f})")
        else:
            parent_score = scores[parent_idx]
            delta = score - parent_score
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            marker = "B" if i == best_idx else ""
            print(f"  C{i}: Child of C{parent_idx} (Score: {score:.4f}, Δ{delta_str}){marker}")
    
    # Calculate prompt length statistics
    prompt_lengths = [len(candidates[i].get('instruction_prompt', '')) for i in range(len(candidates))]
    print(f"\n📏 Prompt length statistics:")
    print(f"  Range: {min(prompt_lengths)} - {max(prompt_lengths)} characters")
    print(f"  Average: {np.mean(prompt_lengths):.1f} characters")
