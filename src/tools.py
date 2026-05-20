'''
Filename: mpifvm2d-py/src/tools.py
Created Date: Sunday, May 17th 2026, 3:14:51 pm
Author: Lixiang Jiang

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''

from typing import List, Set
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from partition import PartitionMesh


def write_tecplot(nodes, elements, val_names, save_pth:str):
    if not save_pth.endswith('.plt'):
        save_pth += '.plt'

    assert nodes.shape[1] == len(val_names), 'Please ensure that the data and name dimensions are consistent.'

    with open(save_pth, 'w') as f:
        f.write('TITLE="Tecplot Unstructured Grid Data"\n')
        f.write('VARIABLES=')
        val_names_str = ''
        for name in val_names:
            val_names_str += name.strip() + ', '
        val_names_str = val_names_str[:-2]
        f.write(val_names_str + '\n')
        f.write('ZONE T="Zone"\n')
        f.write('N={}, E={}, F=FEPOINT, ET=quadrilateral\n'.format(nodes.shape[0], elements.GetTotalNum()))
        f.write('#points:\n')
        for iNode in range(nodes.shape[0]):
            line = ''
            for j in range(nodes.shape[1]):
                line += '{} '.format(nodes[iNode, j])
            f.write(line.strip() + '\n')
        f.write('#elements:\n')
        for iEelem in range(elements.GetTotalNum()):
            if elements.GetNumPart(iEelem) == 4:
                line = ''
                for k in range(elements.GetNumPart(iEelem)):
                    node_idx = elements.GetData(iEelem, k)
                    line += '{} '.format(node_idx + 1)
                f.write(line.strip() + '\n')
            elif elements.GetNumPart(iEelem) == 3:
                line = ''
                for k in range(elements.GetNumPart(iEelem)):
                    node_idx = elements.GetData(iEelem, k)
                    line += '{} '.format(node_idx + 1)
                line += '{} '.format(elements.GetData(iEelem, 0) + 1)
                f.write(line.strip() + '\n')
            else:
                raise ValueError('Unsupported element type with {} nodes.'.format(elements.GetNumPart(iEelem)))
            
def show_sparse_matrix_info(pt_sur_pt:List[Set[int]], n_domain=None, compare_with=None):
    """可视化邻接关系构成的稀疏矩阵（非零元分布）"""
    n = len(pt_sur_pt)
    # 创建 LIL 稀疏矩阵，方便按行赋值
    A = lil_matrix((n, n), dtype=np.int8)
    
    for i, neighbors in enumerate(pt_sur_pt):
        for j in neighbors:
            A[i, j] = 1   # 有邻接关系则置 1
    
    A = A.tocsr()        # 转为 CSR 格式便于绘图

    if compare_with is not None:
        A_compare = lil_matrix((n, n), dtype=np.int8)
        for i, neighbors in enumerate(compare_with):
            for j in neighbors:
                A_compare[i, j] = 1
        A_compare = A_compare.tocsr()
    
    plt.figure(figsize=(8, 8))

    if n_domain is not None:
        plt.axhline(n_domain - 0.5, linestyle="--", linewidth=1, color='Green')
        plt.axvline(n_domain - 0.5, linestyle="--", linewidth=1, color='Green')


    plt.spy(A, markersize=1, color='blue')
    if compare_with is not None:
        plt.spy(A_compare, markersize=1, color='red', alpha=0.5)
    plt.title(f'Sparsity pattern of adjacency matrix ({n}x{n})')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.show()


def visualize_mesh(mesh:PartitionMesh):
    plt.figure()
    for iEdge in range(mesh.nEdges):
        iPoint1, iPoint2 = mesh.edges[iEdge]
        x1, y1 = mesh.coords[iPoint1]
        x2, y2 = mesh.coords[iPoint2]

        line_color = 'black'
        if iPoint1 > mesh.n_point_domain or iPoint2 > mesh.n_point_domain:
            line_color = 'blue'  # 域内边

        plt.plot([x1, x2], [y1, y2], c=line_color)

        # plt.text(0.5*(x1 + x2), 0.5*(y1 + y2), str(iEdge), fontsize=6, color='blue')

    point_color = 'black'
    for iPoint in range(mesh.n_point):
        x, y = mesh.coords[iPoint]
        if mesh.point_halo_layer[iPoint] == 1:
            point_color = 'green'
            plt.plot(x, y, 'o', c=point_color)
        elif mesh.point_halo_layer[iPoint] == 2:
            point_color = 'purple'
            plt.plot(x, y, 'o', c=point_color)

    for iMarker in range(mesh.GetNumMarker()):
        for iBndElem in range(mesh.markers[iMarker].GetNumElem()):
            for iBndNode in range(mesh.markers[iMarker].elem_to_node.GetNumPart(iBndElem)):
                node_idx = mesh.markers[iMarker].elem_to_node.GetData(iBndElem, iBndNode)
                x, y = mesh.coords[node_idx]
                plt.plot(x, y, 'x', c='red')
    
    plt.axis('equal')
    plt.title('Mesh Visualization with Edge Indices')
    plt.show()