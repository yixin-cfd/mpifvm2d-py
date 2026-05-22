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


def ElemCG(elem_type, points):
    """Obtain the centroid of the element and the centroid of the corresponding face"""
    elemInfo = ElemInfo()

    dim = points.shape[1]
    Coord_CG = zeros((1, dim), dtype=float64)
    nNodes = points.shape[0]
    for iNode in range(nNodes):
        Coord_CG += points[iNode]/float(nNodes)

    Coord_FaceElems_CG = zeros((elemInfo.n_faces[elem_type], dim), dtype=float64)

    for iFace in range(elemInfo.n_faces[elem_type]):
        for iNode in range(elemInfo.n_nodes_face[elem_type, iFace]):
            node_idx = elemInfo.faces[elem_type, iFace, iNode]
            Coord_FaceElems_CG[iFace] += points[node_idx]/float(elemInfo.n_nodes_face[elem_type, iFace])

    if False:
        plt.figure()
        for iNode in range(nNodes):
            plt.plot(points[iNode, 0], points[iNode, 1], 'ko')
            plt.text(points[iNode, 0], points[iNode, 1], str(iNode), fontsize=12)
        plt.plot(Coord_CG[0, 0], Coord_CG[0, 1], 'ro', label='Element CG')
        for iFace in range(elemInfo.n_faces[elem_type]):
            plt.plot(Coord_FaceElems_CG[iFace, 0], Coord_FaceElems_CG[iFace, 1], 'go', label='Face Element CG' if iFace == 0 else '')
        plt.legend()
        plt.show()

    return Coord_CG, Coord_FaceElems_CG

def GetMeshCG(mesh:PartitionMesh):
    All_Elem_CG = []
    All_FaceElems_CG = []
    All_Edge_CG = []

    # body elements
    for iElem in range(mesh.elem.GetTotalNum()):
        Coord_CG, Coord_FaceElems_CG = ElemCG(mesh.elem_type[iElem], mesh.coords[_get_csr_row(mesh.elem, iElem)])
        All_Elem_CG.append(Coord_CG)
        All_FaceElems_CG.append(Coord_FaceElems_CG)
    # boundary elements
    All_BndElem_CG = []
    All_FaceBndElems_CG = []
    for iMarker in range(mesh.GetNumMarker()):
        for iBndElem in range(mesh.markers[iMarker].GetNumElem()):
            Coord_CG, Coord_FaceElems_CG = ElemCG(mesh.markers[iMarker].elem_type[iBndElem], \
                                                  mesh.coords[_get_csr_row(mesh.markers[iMarker].elem_to_node, iBndElem)])
            All_BndElem_CG.append(Coord_CG)
            All_FaceBndElems_CG.append(Coord_FaceElems_CG)
    # edge CG
    for iEdge in range(mesh.nEdges):
        iPoint, jPoint = mesh.edges[iEdge]
        Coord_CG = (mesh.coords[iPoint] + mesh.coords[jPoint]) / 2.0
        All_Edge_CG.append(Coord_CG.reshape(1, -1))

    return All_Elem_CG, All_FaceElems_CG, All_BndElem_CG, All_FaceBndElems_CG, All_Edge_CG

def FindEdge(edges, iPoint, jPoint):
    for iEdge, edge in enumerate(edges):
        if (edge[0] == iPoint and edge[1] == jPoint) or (edge[0] == jPoint and edge[1] == iPoint):
            return iEdge
    return -1

def GetControlVolume(mesh:PartitionMesh, All_Elem_CG, All_FaceElems_CG, All_BndElem_CG, All_FaceBndElems_CG):
    nDim = mesh.coords.shape[1]

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

    vertexs = CreateVertex(mesh)
    mesh.vertexs = vertexs
    return mesh

def GetEdge_Normal_2D(Coord_Edge_CG, Coord_Elem_CG):
    assert Coord_Elem_CG.shape[1] == 2 and Coord_Edge_CG.shape[1] == 2, "Only support 2D edge normal calculation."
    # need to carefully determine the direction of the normal vector, the normal must belong to the small number point!!
    normal = zeros(2, dtype=float64)
    normal[0] = Coord_Elem_CG[0, 1] - Coord_Edge_CG[0, 1]
    normal[1] = - (Coord_Elem_CG[0, 0] - Coord_Edge_CG[0, 0])
    return normal
    
def GetVolume_2D(Coord_Edge_CG, Coord_Elem_CG, Coord_Point):
    Coord_Edge_CG = Coord_Edge_CG[0]
    Coord_Elem_CG = Coord_Elem_CG[0]
    
    vec_a = Coord_Elem_CG - Coord_Point
    vec_b = Coord_Edge_CG - Coord_Point
    return 0.5 * abs(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])


def GetEdge_Normal_3D(Coord_Elem_CG, Coord_Edge_CG, Coord_FaceElem_CG):
    assert Coord_Elem_CG.shape[1] == 3 and Coord_Edge_CG.shape[0] == 3 and Coord_FaceElem_CG.shape[0] == 3, "Only support 3D edge normal calculation."
    raise NotImplementedError("3D edge normal calculation is not implemented yet.")

def Metrics(mesh:PartitionMesh):
    """compute edge normals, control volume, etc."""
    All_Elem_CG, All_FaceElems_CG, All_BndElem_CG, All_FaceBndElems_CG, All_Edge_CG = GetMeshCG(mesh)
    nDim = mesh.coords.shape[1]
    #
    All_Edge_Normals = zeros((mesh.nEdges, nDim), dtype=float64)
    All_Control_Volumes = zeros(mesh.n_point, dtype=float64)
    elem_info = ElemInfo()
    for iElem in range(mesh.elem.GetTotalNum()):
        elem_type = mesh.elem_type[iElem]
        for iFace in range(elem_info.n_faces[elem_type]):
            if nDim == 2:
                nEdgeFace = 1       # 2D 网格中每个面只有一条边
            elif nDim == 3:
                nEdgeFace = elem_info.n_nodes_face[elem_type, iFace] # 3D 网格中每个面由多个边组成
            for iEdgesFace in range(nEdgeFace):
                if nDim == 2:
                    face_iPoint = mesh.elem.GetData(iElem, elem_info.faces[elem_type, iFace, 0])
                    face_jPoint = mesh.elem.GetData(iElem, elem_info.faces[elem_type, iFace, 1])
                if nDim == 3:
                    face_iPoint = mesh.elem.GetData(iElem, elem_info.faces[elem_type, iFace, iEdgesFace])
                    face_jPoint = mesh.elem.GetData(iElem, elem_info.faces[elem_type, iFace, (iEdgesFace+1)%nEdgeFace])
                change_face_orientation = False
                change_face_orientation = True if face_iPoint > face_jPoint else False
                iEdge = FindEdge(mesh.edges, face_iPoint, face_jPoint)
                #
                Coord_Edge_CG = All_Edge_CG[iEdge]
                Coord_Elem_CG = All_Elem_CG[iElem]
                Coord_FaceElem_CG = All_FaceElems_CG[iElem][iFace]
                Coord_FaceiPoint = mesh.coords[face_iPoint]
                Coord_FacejPoint = mesh.coords[face_jPoint]
                #
                if False:
                    plt.figure()
                    nodes = mesh.elem.GetNumPart(iElem)
                    elem_nodes = mesh.coords[[mesh.elem.GetData(iElem, k) for k in range(nodes)]]
                    for iNode in range(nodes):
                        plt.plot(elem_nodes[iNode, 0], elem_nodes[iNode, 1], 'ko')
                        plt.text(elem_nodes[iNode, 0], elem_nodes[iNode, 1], str(iNode), fontsize=12)
                    for iNode in range(nodes):
                        for iNeighbor in range(elem_info.n_neighbor_nodes[elem_type, iNode]):
                            neighbor_node_idx = elem_info.neighbor_nodes[elem_type, iNode, iNeighbor]
                            plt.plot([elem_nodes[iNode, 0], elem_nodes[neighbor_node_idx, 0]], [elem_nodes[iNode, 1], elem_nodes[neighbor_node_idx, 1]], 'k-')
                    plt.plot(Coord_Elem_CG[0, 0], Coord_Elem_CG[0, 1], 'ro', label='Element CG')
                    plt.plot(Coord_FaceElem_CG[0], Coord_FaceElem_CG[1], 'go', label='Face Element CG')
                    plt.plot(Coord_Edge_CG[0], Coord_Edge_CG[1], 'bx', label='Edge CG')
                    plt.show()
                if nDim == 2:
                    if change_face_orientation:
                        edge_normal = GetEdge_Normal_2D(Coord_Elem_CG, Coord_Edge_CG)
                    else:
                        edge_normal = GetEdge_Normal_2D(Coord_Edge_CG, Coord_Elem_CG)
                    All_Edge_Normals[iEdge] += edge_normal
                    All_Control_Volumes[face_iPoint] += GetVolume_2D(Coord_Edge_CG, Coord_Elem_CG, Coord_FaceiPoint)
                    All_Control_Volumes[face_jPoint] += GetVolume_2D(Coord_Edge_CG, Coord_Elem_CG, Coord_FacejPoint)
                elif nDim == 3:
                    raise NotImplementedError("3D edge normal calculation is not implemented yet.")
    # check edge normals
    for iEdge in range(mesh.nEdges):
        EPS = 1e-16
        normal = All_Edge_Normals[iEdge]
        norm = np.linalg.norm(normal)
        if norm  == 0.0:
            print('Warning: edge {} has near-zero normal vector.'.format(iEdge))
            All_Edge_Normals[iEdge] = np.array([EPS*EPS]*nDim, dtype=float64)
    return All_Edge_Normals, All_Control_Volumes

def residual(mesh, field):
    """compute residual for FVM solver."""
    pass

def halo_exchange(mesh, field):
    """perform halo exchange for FVM solver."""
    pass