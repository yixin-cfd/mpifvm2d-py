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
    elem_info = ElemInfo()

    adjacency_sets = [set() for _ in range(len(nodes))]

    for iElem in range(elements.GetTotalNum()):
        elem_type = int(elements.elem_types[iElem])
        n_nodes = elements.GetNumPart(iElem)

        if elem_info.n_nodes[elem_type] == 0:
            raise ValueError('Unsupported element type {}.'.format(elem_type))

        for iNode in range(n_nodes):
            iPoint = int(elements.GetData(iElem, iNode))
            n_neighbor_nodes = int(elem_info.n_neighbor_nodes[elem_type, iNode])

            for iNeighbor in range(n_neighbor_nodes):
                jNode = int(elem_info.neighbor_nodes[elem_type, iNode, iNeighbor])
                jPoint = int(elements.GetData(iElem, jNode))
                adjacency_sets[iPoint].add(jPoint)

    adjacency = []
    for iNode in range(len(nodes)):
        adjacency.append(sorted(adjacency_sets[iNode]))

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


