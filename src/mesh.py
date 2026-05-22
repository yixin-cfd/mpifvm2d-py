'''
Created Date: Friday, May 15th 2026, 10:32:58 pm
Author: Lixiang Jiang 

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.


Try to use the Fortran syntax style.!!!
'''
from DS import CSR
from numpy import array, zeros, float64, int32, ones
import numpy as np
import matplotlib.pyplot as plt

max_cgns_type = 50
max_face = 6
max_node_face = 4
max_node_elem = 8



CT2Node = zeros(max_cgns_type, dtype=int32)    # CGNS type to number of nodes
CT2Node[3] = 2    # Line
CT2Node[5] = 3    # Triangle
CT2Node[9] = 4    # Quadrilateral
CT2Node[10] = 4   # Tetrahedral
CT2Node[12] = 8   # Hexahedral


class ElemInfo:
    """
    静态单元拓扑表。

    尽量采用 Fortran 风格：
        elem_type
        n_faces
        n_nodes
        n_nodes_face
        faces
        neighbor_nodes

    局部节点编号采用 0-based。
    """

    def __init__(self):
        self.n_nodes = zeros(max_cgns_type, dtype=int32)
        self.n_faces = zeros(max_cgns_type, dtype=int32)
        self.max_nodes_face = zeros(max_cgns_type, dtype=int32)

        self.n_nodes_face = zeros((max_cgns_type, max_face), dtype=int32)

        # faces[elem_type, i_face, i_node_face]
        self.faces = -ones((max_cgns_type, max_face, max_node_face), dtype=int32)

        # neighbor_nodes[elem_type, i_node, i_neighbor_node]
        self.n_neighbor_nodes = zeros((max_cgns_type, max_node_elem), dtype=int32)
        self.neighbor_nodes = -ones((max_cgns_type, max_node_elem, max_node_elem), dtype=int32)

        self._init_line()
        self._init_triangle()
        self._init_quad()
        self._init_tetra()

    def _init_line(self):
        et = 3

        self.n_nodes[et] = 2
        self.n_faces[et] = 1
        self.max_nodes_face[et] = 1

        # 1D line 的 face 是两个端点
        self.n_nodes_face[et, 0] = 2


        self.faces[et, 0, 0] = 0
        self.faces[et, 1, 0] = 1

        self.n_neighbor_nodes[et, 0] = 1
        self.n_neighbor_nodes[et, 1] = 1

        self.neighbor_nodes[et, 0, 0] = 1
        self.neighbor_nodes[et, 1, 0] = 0

    def _init_triangle(self):
        et = 5

        self.n_nodes[et] = 3
        self.n_faces[et] = 3
        self.max_nodes_face[et] = 2

        self.n_nodes_face[et, 0:3] = [2, 2, 2]

        self.faces[et, 0, 0:2] = [0, 1]
        self.faces[et, 1, 0:2] = [1, 2]
        self.faces[et, 2, 0:2] = [2, 0]

        self.n_neighbor_nodes[et, 0:3] = [2, 2, 2]

        self.neighbor_nodes[et, 0, 0:2] = [1, 2]
        self.neighbor_nodes[et, 1, 0:2] = [2, 0]
        self.neighbor_nodes[et, 2, 0:2] = [0, 1]

    def _init_quad(self):
        et = 9

        self.n_nodes[et] = 4
        self.n_faces[et] = 4
        self.max_nodes_face[et] = 2

        self.n_nodes_face[et, 0:4] = [2, 2, 2, 2]

        self.faces[et, 0, 0:2] = [0, 1]
        self.faces[et, 1, 0:2] = [1, 2]
        self.faces[et, 2, 0:2] = [2, 3]
        self.faces[et, 3, 0:2] = [3, 0]

        self.n_neighbor_nodes[et, 0:4] = [2, 2, 2, 2]

        self.neighbor_nodes[et, 0, 0:2] = [1, 3]
        self.neighbor_nodes[et, 1, 0:2] = [2, 0]
        self.neighbor_nodes[et, 2, 0:2] = [3, 1]
        self.neighbor_nodes[et, 3, 0:2] = [0, 2]

    def _init_tetra(self):
        et = 10

        self.n_nodes[et] = 4
        self.n_faces[et] = 4
        self.max_nodes_face[et] = 3

        self.n_nodes_face[et, 0:4] = [3, 3, 3, 3]

        self.faces[et, 0, 0:3] = [0, 2, 1]
        self.faces[et, 1, 0:3] = [0, 1, 3]
        self.faces[et, 2, 0:3] = [0, 3, 2]
        self.faces[et, 3, 0:3] = [1, 2, 3]

        self.n_neighbor_nodes[et, 0:4] = [3, 3, 3, 3]

        self.neighbor_nodes[et, 0, 0:3] = [1, 2, 3]
        self.neighbor_nodes[et, 1, 0:3] = [0, 2, 3]
        self.neighbor_nodes[et, 2, 0:3] = [0, 1, 3]
        self.neighbor_nodes[et, 3, 0:3] = [0, 1, 2]



def read_SU2_mesh(mesh_pth):
    lines = open(mesh_pth, 'r').readlines()
    
    def read_dim(lines):
        for line in lines:
            if line.strip().startswith('NDIME'):
                return int(line.split()[1])
        raise ValueError('NDIME not found in mesh file.')

    def read_nodes(lines, dim):
        nodes = []
        for i, line in enumerate(lines):
            if line.strip().startswith('NPOIN'):
                num_nodes = int(line.split()[1])
                for j in range(num_nodes):
                    node_line = lines[i + 1 + j].strip()
                    node_coords = list(map(float, node_line.split()[:dim]))
                    nodes.append(node_coords)
                break
        nodes = array(nodes, dtype=float64)
        return nodes
    
    def read_element(lines):
        idx = -1
        num_elements = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('NELEM'):
                num_elements = int(line.split()[1])
                print(f'Number of elements: {num_elements}')
                idx = i
                break
        if idx == -1:
            raise ValueError('NELEM not found in mesh file.')
        
        parts = zeros(num_elements, dtype=int32)
        elem_types = zeros(num_elements, dtype=int32)
        for j in range(num_elements):
            elem_types[j] = int(lines[idx + 1 + j].strip().split()[0])
            parts[j] = CT2Node[elem_types[j]]

        csr = CSR(parts, dtype=int32)
        for j in range(num_elements):
            line = lines[idx + 1 + j].strip().split()
            node_indices = list(map(int, line[1:1 + parts[j]]))
            for k in range(parts[j]):
                csr.SetData(j, k, node_indices[k])
        csr.AddAttr(elem_types=elem_types)
        return csr
    
    def read_boundary(lines):
        """
        读取 SU2 网格文件中的边界标记 (NMARK)
        返回: list of dict, 每个 dict 包含:
            'name'  : 边界标记名 (如 'AIRFOIL')
            'elems' : CSR 对象，存储边界元素的节点连接，并附带 elem_types 属性
        """
        boundaries = []
        idx = -1
        # 定位 NMARK 行
        for i, line in enumerate(lines):
            if line.strip().startswith('NMARK'):
                nmark = int(line.split()[1])
                idx = i + 1
                break
        if idx == -1:
            raise ValueError('NMARK not found in mesh file.')

        for _ in range(nmark):
            # 跳过可能的空行，找到 MARKER_TAG
            line = lines[idx].strip()
            while not line.startswith('MARKER_TAG'):
                idx += 1
                line = lines[idx].strip()
            marker_name = line.split('=')[1].strip()
            idx += 1

            # 找到 MARKER_ELEMS
            line = lines[idx].strip()
            while not line.startswith('MARKER_ELEMS'):
                idx += 1
                line = lines[idx].strip()
            nelem = int(line.split('=')[1].strip())
            idx += 1

            # 记录每个边界元素的节点数（类型决定）
            parts = zeros(nelem, dtype=int32)
            elem_types = zeros(nelem, dtype=int32)

            for j in range(nelem):
                line = lines[idx + j].strip().split()
                elem_type = int(line[0])
                num_nodes = CT2Node[elem_type]
                parts[j] = num_nodes
                elem_types[j] = elem_type

            # 构建 CSR 存储连接
            csr = CSR(parts, dtype=int32)
            for j in range(nelem):
                line = lines[idx + j].strip().split()
                node_indices = [int(x) for x in line[1:1 + parts[j]]]
                for k, node in enumerate(node_indices):
                    csr.SetData(j, k, node)

            csr.AddAttr(elem_types=elem_types)
            boundaries.append({'name': marker_name, 'elems': csr})

            idx += nelem

        return boundaries

    dim = read_dim(lines)
    nodes = read_nodes(lines, dim)
    elements = read_element(lines)
    boundaries = read_boundary(lines)

    print(f'The {mesh_pth} dimension: {dim}')
    print(f'The {mesh_pth} nodes: {len(nodes)}')
    print(f'The {mesh_pth} elements: {elements.GetTotalNum()}')
    print(f'First 5 nodes: {nodes[:5]}')
    print(f'First 5 elements: {elements.data[:5]}')
    print(f'different element types: {set(elements.elem_types)}')
    print(f'Boundary conditions: {[b["name"] for b in boundaries]}')

    return nodes, elements, boundaries


def plot_mesh(mesh_data):
    nodes, elements, boundaries = mesh_data

    plt.figure()
    for e in range(elements.GetTotalNum()):
        elem_type = elements.elem_types[e]
        node_indices = [elements.GetData(e, k) for k in range(elements.GetNumPart(e))]
        elem_nodes = nodes[node_indices]
        if elem_type == 5:  # Triangle
            plt.fill(elem_nodes[:, 0], elem_nodes[:, 1], edgecolor='k', fill=False)
        elif elem_type == 9:  # Quadrilateral
            plt.fill(elem_nodes[:, 0], elem_nodes[:, 1], edgecolor='k', fill=False)

    for boundary in boundaries:
        for e in range(boundary['elems'].GetTotalNum()):
            elem_type = boundary['elems'].elem_types[e]
            node_indices = [boundary['elems'].GetData(e, k) for k in range(boundary['elems'].GetNumPart(e))]
            elem_nodes = nodes[node_indices]
            if elem_type == 3:  # Line
                plt.plot(elem_nodes[:, 0], elem_nodes[:, 1], 'r-')

    plt.axis('equal')
    plt.title('Mesh Visualization')
    plt.show()


if __name__ == "__main__":

    mesh_pth = r'mesh/mesh_RAE2822_turb.su2'
    res = read_SU2_mesh(mesh_pth)

    plot_mesh(res)
