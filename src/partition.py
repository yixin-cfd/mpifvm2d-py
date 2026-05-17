'''
Filename: mpifvm2d-py/src/partition.py
Created Date: Saturday, May 16th 2026, 5:20:43 pm
Author: Lixiang Jiang

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''

from mesh import ElemInfo, CT2Node, read_SU2_mesh
import pymetis
from numpy import array, zeros, int32, int64, float64
import numpy as np
from mpi4py import MPI

from DS import CSR, MarkerData, CommPattern, PartitionMesh
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
    edge_cuts, parts = pymetis.part_graph(n_partitions, adjacency=adjacency)
    return np.array(parts, dtype=int32)


def _get_csr_row(csr, iPart):
    return [int(csr.GetData(iPart, idx)) for idx in range(csr.GetNumPart(iPart))]


def _build_csr_from_rows(rows, dtype=int32):
    parts = zeros(len(rows), dtype=int32)
    for iPart, row in enumerate(rows):
        parts[iPart] = len(row)

    csr = CSR(parts, dtype=dtype)
    for iPart, row in enumerate(rows):
        for idx, val in enumerate(row):
            csr.SetData(iPart, idx, val)

    return csr


def _build_point_surrounding_elements(elements, n_point):
    point_to_elem = [[] for _ in range(n_point)]

    for iElem in range(elements.GetTotalNum()):
        elem_nodes = _get_csr_row(elements, iElem)
        for iPoint in elem_nodes:
            point_to_elem[iPoint].append(iElem)

    return point_to_elem


def _build_marker_data(name, global_bnd_ids, bnd_elems, global_to_local_point):
    global_bnd_ids = sorted(global_bnd_ids)
    rows = []
    elem_type = zeros(len(global_bnd_ids), dtype=int32)
    local_to_global_bnd_elem = zeros(len(global_bnd_ids), dtype=int64)

    for iLocalElem, iGlobalElem in enumerate(global_bnd_ids):
        global_nodes = _get_csr_row(bnd_elems, iGlobalElem)
        rows.append([global_to_local_point[iPoint] for iPoint in global_nodes])
        elem_type[iLocalElem] = int(bnd_elems.elem_types[iGlobalElem])
        local_to_global_bnd_elem[iLocalElem] = iGlobalElem

    elem_to_node = _build_csr_from_rows(rows, dtype=int32)
    elem_to_node.AddAttr(elem_types=elem_type)

    return MarkerData(name, elem_to_node, elem_type, local_to_global_bnd_elem)


def build_partition_meshes(mesh_data, point_color, n_partitions, n_halo_layer=2):
    nodes, elements, boundaries = mesh_data
    point_color = np.array(point_color, dtype=int32)

    if len(point_color) != len(nodes):
        raise ValueError('point_color size must match number of mesh nodes.')

    if np.any(point_color < 0) or np.any(point_color >= n_partitions):
        raise ValueError('point_color contains invalid partition ids.')

    if n_halo_layer < 1:
        raise ValueError('n_halo_layer must be at least 1 for FVM partition meshes.')

    n_dim = nodes.shape[1]
    point_to_elem = _build_point_surrounding_elements(elements, len(nodes))
    local_elem_ids = [set() for _ in range(n_partitions)]
    local_point_layers = [dict() for _ in range(n_partitions)]
    local_marker_elem_ids = [[set() for _ in boundaries] for _ in range(n_partitions)]

    # 构造 owned 点、halo 点、局部单元
    for iRank in range(n_partitions):
        owned_points = [iPoint for iPoint in range(len(nodes)) if int(point_color[iPoint]) == iRank]    # 当前 rank 拥有的点
        for iPoint in owned_points:
            local_point_layers[iRank][iPoint] = 0

        frontier = set(owned_points)        # 初始化扩展边界 frontier
        for iLayer in range(1, n_halo_layer + 1):
            next_frontier = set()
            for iPoint in frontier:         # 按 halo 层数向外扩展 
                for iElem in point_to_elem[iPoint]:
                    local_elem_ids[iRank].add(iElem)
                    elem_nodes = _get_csr_row(elements, iElem)
                    for jPoint in elem_nodes:
                        if jPoint not in local_point_layers[iRank]:
                            local_point_layers[iRank][jPoint] = iLayer
                            next_frontier.add(jPoint)
            frontier = next_frontier

    # 如果某个边界单元的任意一个点出现在当前 rank 的局部点集合里，那么这个边界单元也要加入当前 rank 的 marker 数据
    for iRank in range(n_partitions):
        local_point_ids = set(local_point_layers[iRank].keys())
        for iMarker, boundary in enumerate(boundaries):
            bnd_elems = boundary['elems']
            for iBndElem in range(bnd_elems.GetTotalNum()):
                bnd_nodes = _get_csr_row(bnd_elems, iBndElem)
                if any(iPoint in local_point_ids for iPoint in bnd_nodes):
                    local_marker_elem_ids[iRank][iMarker].add(iBndElem)
                    for iPoint in bnd_nodes:
                        if iPoint not in local_point_layers[iRank]:
                            local_point_layers[iRank][iPoint] = n_halo_layer    # marker as maximum halo layer for boundary points
                            local_point_ids.add(iPoint)

    part_meshes = []

    for iRank in range(n_partitions):
        owned_points = [iPoint for iPoint, iLayer in local_point_layers[iRank].items() if iLayer == 0]
        ghost_points = [iPoint for iPoint, iLayer in local_point_layers[iRank].items() if iLayer > 0]
        owned_points.sort()
        ghost_points.sort(key=lambda iPoint: (local_point_layers[iRank][iPoint], iPoint))

        local_to_global_point = np.array(owned_points + ghost_points, dtype=int64)
        global_to_local_point = {int(iGlobal): iLocal for iLocal, iGlobal in enumerate(local_to_global_point)}
        point_halo_layer = np.array([local_point_layers[iRank][int(iGlobal)] for iGlobal in local_to_global_point],
                                    dtype=int32)

        part_mesh = PartitionMesh(rank=iRank, size=n_partitions, n_dim=n_dim)
        part_mesh.n_halo_layer = n_halo_layer
        part_mesh.n_point_domain = len(owned_points)
        part_mesh.n_point_ghost = len(ghost_points)
        part_mesh.n_point = len(local_to_global_point)
        part_mesh.local_to_global_point = local_to_global_point
        part_mesh.global_to_local_point = global_to_local_point
        part_mesh.coords = nodes[local_to_global_point].copy()
        part_mesh.point_color = point_color[local_to_global_point].copy()
        part_mesh.point_halo_layer = point_halo_layer

        elem_ids = sorted(local_elem_ids[iRank])
        elem_rows = []
        elem_type = zeros(len(elem_ids), dtype=int32)
        local_to_global_elem = zeros(len(elem_ids), dtype=int64)

        for iLocalElem, iGlobalElem in enumerate(elem_ids):
            global_nodes = _get_csr_row(elements, iGlobalElem)
            elem_rows.append([global_to_local_point[iPoint] for iPoint in global_nodes])
            elem_type[iLocalElem] = int(elements.elem_types[iGlobalElem])
            local_to_global_elem[iLocalElem] = iGlobalElem

        elem_to_node = _build_csr_from_rows(elem_rows, dtype=int32)
        elem_to_node.AddAttr(elem_types=elem_type)

        part_mesh.n_elem = len(elem_ids)
        part_mesh.elem_to_node = elem_to_node
        part_mesh.elem_type = elem_type
        part_mesh.local_to_global_elem = local_to_global_elem

        for iMarker, boundary in enumerate(boundaries):
            marker = _build_marker_data(boundary['name'], local_marker_elem_ids[iRank][iMarker],
                                        boundary['elems'], global_to_local_point)
            part_mesh.AddMarker(marker)

        part_meshes.append(part_mesh)

    return part_meshes


def build_comm_patterns(part_meshes, point_color):
    n_partitions = len(part_meshes)
    point_color = np.array(point_color, dtype=int32)

    send_global = [[set() for _ in range(n_partitions)] for _ in range(n_partitions)]
    recv_global = [[set() for _ in range(n_partitions)] for _ in range(n_partitions)]

    for iRank, part_mesh in enumerate(part_meshes):
        for iLocalPoint in range(part_mesh.n_point_domain, part_mesh.n_point):
            iGlobalPoint = int(part_mesh.local_to_global_point[iLocalPoint])
            owner = int(point_color[iGlobalPoint])
            recv_global[iRank][owner].add(iGlobalPoint)
            send_global[owner][iRank].add(iGlobalPoint)

    for iRank, part_mesh in enumerate(part_meshes):
        send_rows = []
        recv_rows = []
        send_global_rows = []
        recv_global_rows = []

        for jRank in range(n_partitions):
            send_points = sorted(send_global[iRank][jRank])
            recv_points = sorted(recv_global[iRank][jRank])

            send_rows.append([part_mesh.global_to_local_point[iPoint] for iPoint in send_points])
            recv_rows.append([part_mesh.global_to_local_point[iPoint] for iPoint in recv_points])
            send_global_rows.append(send_points)
            recv_global_rows.append(recv_points)

        send_nodes = _build_csr_from_rows(send_rows, dtype=int32)
        recv_nodes = _build_csr_from_rows(recv_rows, dtype=int32)
        send_point_global = _build_csr_from_rows(send_global_rows, dtype=int64)
        recv_point_global = _build_csr_from_rows(recv_global_rows, dtype=int64)

        send_ranks = []
        recv_ranks = []
        for jRank in range(n_partitions):
            if send_nodes.GetNumPart(jRank) > 0:
                send_ranks.append(jRank)
            if recv_nodes.GetNumPart(jRank) > 0:
                recv_ranks.append(jRank)

        part_mesh.comm = CommPattern(send_nodes=send_nodes,
                                     recv_nodes=recv_nodes,
                                     send_ranks=np.array(send_ranks, dtype=int32),
                                     recv_ranks=np.array(recv_ranks, dtype=int32),
                                     send_point_global=send_point_global,
                                     recv_point_global=recv_point_global)

    return part_meshes


def check_partition_meshes(part_meshes, point_color):
    point_color = np.array(point_color, dtype=int32)

    for part_mesh in part_meshes:
        if part_mesh.n_point != len(part_mesh.local_to_global_point):
            raise ValueError('Rank {} has inconsistent point count.'.format(part_mesh.rank))

        for iLocalPoint in range(part_mesh.n_point):
            iGlobalPoint = int(part_mesh.local_to_global_point[iLocalPoint])
            owner = int(point_color[iGlobalPoint])
            if int(part_mesh.point_color[iLocalPoint]) != owner:
                raise ValueError('Rank {} has inconsistent point color.'.format(part_mesh.rank))
            if iLocalPoint < part_mesh.n_point_domain and owner != part_mesh.rank:
                raise ValueError('Rank {} has non-owned point in domain range.'.format(part_mesh.rank))
            if iLocalPoint >= part_mesh.n_point_domain and owner == part_mesh.rank:
                raise ValueError('Rank {} has owned point in ghost range.'.format(part_mesh.rank))

        if len(part_mesh.elem_to_node.data) > 0:
            if np.min(part_mesh.elem_to_node.data) < 0 or np.max(part_mesh.elem_to_node.data) >= part_mesh.n_point:
                raise ValueError('Rank {} has invalid element connectivity.'.format(part_mesh.rank))

        for marker in part_mesh.markers:
            if marker.GetNumElem() == 0:
                continue
            if np.min(marker.elem_to_node.data) < 0 or np.max(marker.elem_to_node.data) >= part_mesh.n_point:
                raise ValueError('Rank {} has invalid boundary connectivity.'.format(part_mesh.rank))

    for part_mesh in part_meshes:
        if part_mesh.comm is None:
            continue
        for jRank in part_mesh.comm.send_ranks:
            jRank = int(jRank)
            send_globals = [int(part_mesh.comm.send_point_global.GetData(jRank, i))
                            for i in range(part_mesh.comm.send_point_global.GetNumPart(jRank))]
            other = part_meshes[jRank]
            recv_globals = [int(other.comm.recv_point_global.GetData(part_mesh.rank, i))
                            for i in range(other.comm.recv_point_global.GetNumPart(part_mesh.rank))]
            if send_globals != recv_globals:
                raise ValueError('Communication mismatch between rank {} and rank {}.'.format(part_mesh.rank, jRank))

    return True


def parts_info(parts):
    parts = list(parts)
    unique_parts = set(parts)
    part_counts = {part: parts.count(part) for part in unique_parts}
    return part_counts

if __name__ == "__main__":
    mesh_pth = r'mesh/mesh_RAE2822_turb.su2'
    res = read_SU2_mesh(mesh_pth)

    n_partitions = 4

    adjacency = build_adjacency(res)
    print(adjacency[:5])
    point_color = graph_partition(adjacency, n_partitions=n_partitions)
    print(point_color[:20])
    part_counts = parts_info(point_color)
    print(part_counts)

    part_meshes = build_partition_meshes(res, point_color, n_partitions=n_partitions, n_halo_layer=2)
    build_comm_patterns(part_meshes, point_color)
    check_partition_meshes(part_meshes, point_color)
    for part_mesh in part_meshes:
        print('Rank {}: n_point={}, n_domain={}, n_ghost={}, n_elem={}, send_ranks={}, recv_ranks={}'.format(
            part_mesh.rank, part_mesh.n_point, part_mesh.n_point_domain, part_mesh.n_point_ghost,
            part_mesh.n_elem, list(part_mesh.comm.send_ranks), list(part_mesh.comm.recv_ranks)))

    nodes, elements, boundaries = res

    nodes = array(nodes, dtype=float64)
    point_color_plot = array(point_color, dtype=float64).reshape(-1, 1)
    nodes = np.concatenate((nodes, point_color_plot), axis=1)

    val_names = ['X', 'Y', 'Parts']
    save_pth = r'output/mesh_RAE2822_turb_partition.plt'
    write_tecplot(nodes, elements, val_names, save_pth)


