import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# 读取CSV文件
df = pd.read_csv('roads_manhattan_modify2.csv')  # 替换为你的CSV文件名

# 创建节点映射字典，用于存储相同位置的节点映射关系
node_mapping = {}
# 创建位置字典，用于检查重复位置
position_dict = {}

# 首先建立节点映射关系
for _, row in df.iterrows():
    node1, node2 = row[3], row[4]
    lat1, lon1 = round(row[5], 3), round(row[6], 3)
    lat2, lon2 = round(row[7], 3), round(row[8], 3)
    
    # 处理第一个节点
    pos1 = (lon1, lat1)
    if pos1 in position_dict:
        node_mapping[node1] = position_dict[pos1]
    else:
        position_dict[pos1] = node1
        node_mapping[node1] = node1
    
    # 处理第二个节点
    pos2 = (lon2, lat2)
    if pos2 in position_dict:
        node_mapping[node2] = position_dict[pos2]
    else:
        position_dict[pos2] = node2
        node_mapping[node2] = node2

# 创建一个新的无向图
G = nx.Graph()

# 添加节点和边，使用映射后的节点ID
for pos, node in position_dict.items():
    G.add_node(node, pos=pos)

# 添加边，注意使用映射后的节点ID
for _, row in df.iterrows():
    node1, node2 = row[3], row[4]
    mapped_node1 = node_mapping[node1]
    mapped_node2 = node_mapping[node2]
    
    # 只在两个节点不同时添加边
    if mapped_node1 != mapped_node2:
        G.add_edge(mapped_node1, mapped_node2)

# 创建新的编号映射
old_nodes = list(G.nodes())
new_mapping = {old_node: i+1 for i, old_node in enumerate(old_nodes)}

# 使用nx.relabel_nodes重新编号
G = nx.relabel_nodes(G, new_mapping)

# 更新位置信息
pos = nx.get_node_attributes(G, 'pos')

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制边
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.6)

# 添加节点标签，设置小字体
nx.draw_networkx_labels(G, pos, font_size=4)

# 设置图形参数
plt.title("Graph Structure")
plt.axis('on')
plt.grid(True)
plt.savefig('manhattan.pdf')

with open('manhattan.gpickle', 'wb') as f:
    pickle.dump(G, f)


