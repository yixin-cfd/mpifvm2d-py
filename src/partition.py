'''
Filename: mpifvm2d-py/src/partition.py
Created Date: Saturday, May 16th 2026, 5:20:43 pm
Author: Lixiang Jiang

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''

from mesh import ElemInfo, CT2Node, read_SU2_mesh
import pymetis
from numpy import array, zeros, int32, float64
import numpy as np
from mpi4py import MPI

from DS import CSR
from tools import write_tecplot


def build_adjacency(mesh_data):
    nodes, elements, boundaries = mesh_data

    # count the number of elements of surrounding each node
    parts = zeros(len(nodes), dtype=int32)  
    for iEelem in range(elements.GetTotalNum()):
        for k in range(elements.GetNumPart(iEelem)):
            node_idx = elements.GetData(iEelem, k)
            parts[node_idx] += 1
    sur_elems = CSR(parts, dtype=int32)     # sur elem of each node

    parts = zeros(len(nodes), dtype=int32)
    for iEelem in range(elements.GetTotalNum()):
        for k in range(elements.GetNumPart(iEelem)):
            node_idx = elements.GetData(iEelem, k)
            idx = parts[node_idx]
            sur_elems.SetData(node_idx, idx, iEelem)
            parts[node_idx] += 1

    # get pt surrouding pt
    adjacency = []
    for iNode in range(len(nodes)):
        num_sur_elems = sur_elems.GetNumPart(iNode)
        sur_nodes = set()
        for i in range(num_sur_elems):
            iEelem = sur_elems.GetData(iNode, i)
            for k in range(elements.GetNumPart(iEelem)):
                node_idx = elements.GetData(iEelem, k)
                if node_idx != iNode:
                    sur_nodes.add(node_idx)
        sur_nodes = [int (idx) for idx in list(sur_nodes)]
        adjacency.append(sur_nodes)
    #
    
    return adjacency

def graph_partition(adjacency, n_partitions):
    # 使用 PyMetis 进行图划分
    edge_cuts, parts = pymetis.part_graph(n_partitions, adjacency=adjacency)
    return parts

def parts_info(parts):
    unique_parts = set(parts)
    part_counts = {part: parts.count(part) for part in unique_parts}
    return part_counts

if __name__ == "__main__":
    mesh_pth = r'mesh/mesh_RAE2822_turb.su2'
    res = read_SU2_mesh(mesh_pth)

    adjacency = build_adjacency(res)
    print(adjacency[:5])
    parts = graph_partition(adjacency, n_partitions=4)
    print(parts[:20])
    part_counts = parts_info(parts)
    print(part_counts)

    nodes, elements, boundaries = res

    nodes = array(nodes, dtype=float64)
    parts = array(parts, dtype=float64).reshape(-1, 1)
    nodes = np.concatenate((nodes, parts), axis=1)

    val_names = ['X', 'Y', 'Parts']
    save_pth = r'output/mesh_RAE2822_turb_partition.plt'
    write_tecplot(nodes, elements, val_names, save_pth)


