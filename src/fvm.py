'''
Created Date: Tuesday, May 19th 2026, 4:53:49 pm
Author: Lixiang Jiang 

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''
from partition import (_build_point_surrounding_elements)
from DS import PartitionMesh, _get_csr_row
from numpy import array, zeros, float64, int32, ones
from mesh import ElemInfo, CT2Node
from tools import show_sparse_matrix_info
from typing import List, Set
import matplotlib.pyplot as plt
import numpy as np

def GetPtSurPt(mesh:PartitionMesh):
    elems_sur_point = _build_point_surrounding_elements(mesh.elem, mesh.n_point)
    pt_sur_pt = [set() for _ in range(mesh.n_point)]
    elem_info = ElemInfo()
    for iPoint in range(mesh.n_point):
        elems = elems_sur_point[iPoint]
        for iElem in elems:
            elem_nodes = mesh.elem.GetNumPart(iElem)
            elem_type = mesh.elem_type[iElem]
            for iElemNode in range(elem_nodes):
                node_idx = mesh.elem.GetData(iElem, iElemNode)
                if node_idx == iPoint:
                    for iNeighbor in range(elem_info.n_neighbor_nodes[elem_type, iElemNode]):
                        neighbor_node_idx = elem_info.neighbor_nodes[elem_type, iElemNode, iNeighbor]
                        pt_sur_pt[iPoint].add(mesh.elem.GetData(iElem, neighbor_node_idx))
    return pt_sur_pt


def CreateEdge(pt_sur_pt:List[Set[int]]):
    edges = []
    for iPoint, neighbors in enumerate(pt_sur_pt):
        for neighbor in neighbors:
            if neighbor > iPoint: # 避免重复添加边
                edges.append((iPoint, neighbor))
    return edges

def CreateVertex(mesh:PartitionMesh):
    vertexs = []
    for iMarker in range(mesh.GetNumMarker()):
        cur_vertexs = []
        for iBndElem in range(mesh.markers[iMarker].GetNumElem()):
            for iBndNode in range(mesh.markers[iMarker].elem_to_node.GetNumPart(iBndElem)):
                node_idx = mesh.markers[iMarker].elem_to_node.GetData(iBndElem, iBndNode)
                if node_idx < mesh.n_point_domain: # 只处理域内节点
                    cur_vertexs.append(node_idx)
        vertexs.append(cur_vertexs)
    return vertexs


def RCM(pt_sur_pt:List[Set[int]], mesh:PartitionMesh):
    nPointDomain = mesh.n_point_domain
    """Reverse Cuthill-McKee ordering for better cache performance."""
    Result = []
    Queue = []
    AuxQueue = []

    IsQueued = zeros(len(pt_sur_pt), dtype=bool)    # 创建标记数组,记录节点是否已被处理
    for i in range(len(pt_sur_pt)):
        IsQueued[i] = False

    MinDegree = len(pt_sur_pt[0]); AddPoint = 0 # 初始化最小度数和起始节点
    for i in range(1, len(pt_sur_pt)):
        degree = len(pt_sur_pt[i])
        if degree < MinDegree and i < nPointDomain: 
            for neighbor in pt_sur_pt[i]:
                if neighbor >= nPointDomain: # 尽量选择域内节点作为起始点
                    continue
            MinDegree = degree
            AddPoint = i
    # print('Starting RCM with point {} having minimum degree {}.'.format(AddPoint, MinDegree))
    
    Result.append(AddPoint); IsQueued[AddPoint] = True # 将起始节点加入结果列表并标记为已处理

    while True:
        # 将所有相邻节点按其度数递增的顺序添加到队列中,并检查该元素是否已存在于队列中    
        AuxQueue.clear()
        for iNode in range(len(pt_sur_pt[AddPoint])):
            AdjPoint = list(pt_sur_pt[AddPoint])[iNode]
            if not IsQueued[AdjPoint] and AdjPoint < nPointDomain: # 只处理域内节点
                AuxQueue.append(AdjPoint)

        if len(AuxQueue) > 0:
            AuxQueue.sort(key=lambda x: len(pt_sur_pt[x])) # 按度数排序

            Queue.extend(AuxQueue) # 将排序后的相邻节点添加到队列中 
            for point in AuxQueue:
                IsQueued[point] = True # 标记这些节点为已处理

        if len(Queue) > 0:
            AddPoint = Queue.pop(0) # 从队列中取出下一个节点
            Result.append(AddPoint) # 将该节点加入结果列表

        if len(Queue) == 0:
            break

    for iPoint in range(nPointDomain):
        if not IsQueued[iPoint]: # 如果有未处理的域内节点,则将其加入结果列表
            Result.append(iPoint)
    
    Result.reverse()    # 反转得到RCM顺序

    for iPoint in range(nPointDomain, len(pt_sur_pt)):
        Result.append(iPoint) # 添加 MPI 节点

    # points
    AuxCoord = mesh.coords.copy()
    AuxGlobalIndex = mesh.local_to_global_point.copy()
    mesh.local_to_global_point = np.array([AuxGlobalIndex[i] for i in Result], dtype=int32)
    mesh.coords = np.array([AuxCoord[i] for i in Result], dtype=float64)
    for iPoint in range(mesh.n_point):
        mesh.global_to_local_point[mesh.local_to_global_point[iPoint]] = iPoint
    # connectivity
    InvResult = np.zeros_like(Result, dtype=int32)
    for iPoint in range(len(Result)):
        InvResult[Result[iPoint]] = iPoint
    for iElem in range(mesh.n_elem):
        for iElemNode in range(mesh.elem.GetNumPart(iElem)):
            node_idx = mesh.elem.GetData(iElem, iElemNode)
            new_node_idx = InvResult[node_idx]
            mesh.elem.SetData(iElem, iElemNode, new_node_idx)

    for iMarker in range(mesh.GetNumMarker()):
        for iBndElem in range(mesh.markers[iMarker].GetNumElem()):
            for iBndNode in range(mesh.markers[iMarker].elem_to_node.GetNumPart(iBndElem)):
                node_idx = mesh.markers[iMarker].elem_to_node.GetData(iBndElem, iBndNode)
                new_node_idx = InvResult[node_idx]
                mesh.markers[iMarker].elem_to_node.SetData(iBndElem, iBndNode, new_node_idx)

    return mesh


def build_fvm_struct(args, mesh:PartitionMesh,  RCM_ordering=False):
    """construct edges and vertices(boundary) data structure for FVM solver."""
    pt_sur_pt = GetPtSurPt(mesh)

    #
    if RCM_ordering:
        mesh = RCM(pt_sur_pt, mesh)
        # print('New point order after RCM:', new_order)
        # print('New point order after RCM:', len(new_order), 'points reordered by RCM. domain points: ', mesh.n_point_domain, 'total points: ', mesh.n_point)
        
        new_pt_sur_pt = GetPtSurPt(mesh)
        if args.RCM_show:
            show_sparse_matrix_info(pt_sur_pt, n_domain=mesh.n_point_domain, compare_with=new_pt_sur_pt)
        edges = CreateEdge(new_pt_sur_pt)
    else:
        if args.RCM_show:
            show_sparse_matrix_info(pt_sur_pt, n_domain=mesh.n_point_domain)
        edges = CreateEdge(pt_sur_pt)

    mesh.edges = edges
    mesh.nEdges = len(edges)
    return edges, None

def Metrics(mesh):
    """compute edge normals, control volume, etc."""
    pass

def residual(mesh, field):
    """compute residual for FVM solver."""
    pass

def halo_exchange(mesh, field):
    """perform halo exchange for FVM solver."""
    pass