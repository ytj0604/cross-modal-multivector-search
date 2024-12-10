import matplotlib.pyplot as plt
import numpy as np
import os

file_name = '/mnt/pvse/results/aggregated_results.txt'
# Read data
with open(file_name, 'r') as f:
    data_str = f.read()

# Process data
lines = data_str.strip().split('\n')

data = {}

current_type = None
current_degree = None

for line in lines:
    line = line.strip()
    if line.startswith('Index File:'):
        # Extract type and degree
        index_file = line.split(':')[1].strip()
        filename = os.path.basename(index_file)
        name_parts = filename.split('_')
        if len(name_parts) >= 2:
            current_type = name_parts[0]
            degree_part = name_parts[-1]
            degree_str = degree_part.split('.')[0]
            current_degree = int(degree_str)
        else:
            current_type = None
            current_degree = None
    elif line.startswith('L_pq:'):
        # Parse L_pq, QPS, Recall@10
        parts = line.split(', ')
        data_dict = {}
        for part in parts:
            key, value = part.split(': ')
            key = key.strip()
            value = value.strip()
            data_dict[key] = value
        # Convert values to appropriate types
        L_pq = int(data_dict['L_pq'])
        QPS = float(data_dict['QPS'])
        Recall = float(data_dict['Recall@100'])
        # Store in data
        if current_type is not None and current_degree is not None:
            data.setdefault(current_type, {})
            data[current_type].setdefault(current_degree, [])
            data[current_type][current_degree].append({
                'L_pq': L_pq,
                'QPS': QPS,
                'Recall': Recall
            })

# Plotting
import matplotlib.cm as cm

# For consistent colors and styles
degree_list = sorted({degree for type_data in data.values() for degree in type_data})
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', 'D', '^', 'v', 'p', 'h', '*']
num_styles = len(line_styles)
num_markers = len(markers)

# Assign colors, styles, and markers to degrees
colors = cm.tab20(np.linspace(0, 1, len(degree_list)))  # Use a distinct colormap
style_map = {
    degree: (colors[i % len(colors)], line_styles[i % num_styles], markers[i % num_markers])
    for i, degree in enumerate(degree_list)
}

# For each type (i2i, t2t, t2i, i2t)
# for current_type in data.keys():
#     plt.figure(figsize=(16, 10))
#     plt.title(f'{current_type}', fontsize=20)
#     plt.xlabel('Recall@10 (Log Scale)', fontsize=16)
#     plt.ylabel('QPS', fontsize=16)
#     degrees = sorted(data[current_type].keys())
#     for degree in degrees:
#         color, line_style, marker = style_map[degree]
#         degree_data = data[current_type][degree]
#         degree_data.sort(key=lambda x: x['Recall'])
#         Recall_list = [item['Recall'] for item in degree_data]
#         QPS_list = [item['QPS'] for item in degree_data]
#         L_pq_list = [item['L_pq'] for item in degree_data]
#         # Plot points and lines
#         plt.plot(Recall_list, QPS_list, marker=marker, linestyle=line_style, color=color,
#                  label=f'Degree {degree}', linewidth=2, markersize=10)
#         # Annotate each point with L_pq
#         for Recall, QPS, L_pq in zip(Recall_list, QPS_list, L_pq_list):
#             plt.text(Recall, QPS, f'L_pq={L_pq}', fontsize=12)
#     plt.xscale('log')  # Apply log scale to x-axis
#     plt.legend(fontsize=14)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'{current_type}.png', dpi=300)
#     plt.close()

for current_type in data.keys():
    plt.figure(figsize=(16, 10))
    plt.title(f'{current_type}', fontsize=20)
    plt.xlabel('1 - Recall@10 (Log Scale)', fontsize=16)
    plt.ylabel('QPS', fontsize=16)
    degrees = sorted(data[current_type].keys())
    for degree in degrees:
        color, line_style, marker = style_map[degree]
        degree_data = data[current_type][degree]
        degree_data.sort(key=lambda x: x['Recall'])
        Recall_list = [item['Recall'] for item in degree_data]
        QPS_list = [item['QPS'] for item in degree_data]
        L_pq_list = [item['L_pq'] for item in degree_data]
        
        # Compute Error Rate
        Error_list = [max(1e-10, 1 - Recall) for Recall in Recall_list]
        
        # Plot points and lines using Error Rate
        plt.plot(Error_list, QPS_list, marker=marker, linestyle=line_style, color=color,
                 label=f'Degree {degree}', linewidth=2, markersize=10)
        # Annotate each point with L_pq and Recall
        # for Error, QPS, L_pq, Recall in zip(Error_list, QPS_list, L_pq_list, Recall_list):
        #     plt.text(Error, QPS, f'L_pq={L_pq}\nRecall={Recall:.4f}', fontsize=12)
    plt.xscale('log')  # Apply log scale to x-axis
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{current_type}.png', dpi=300)
    plt.close()
