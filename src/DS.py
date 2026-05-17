'''
Filename: /home/yixin/Data/Code/python/mpifvm2d-py/src/DS.py
Created Date: Friday, May 15th 2026, 7:57:14 pm
Author: Lixiang Jiang 

All rights reserved.

Try to use the Fortran syntax style.!!!
'''
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import numpy as np
from numpy import empty, array, float64, zeros


class CSR:
    """CSR 格式数据存储
        只需存放实际数据和每项数据的起始位置即可，节省空间
    """
    def __init__(self, parts:np.ndarray, dtype=float64):
        NumData = np.sum(parts)
        N_part = len(parts)
        self.data = empty(NumData, dtype=dtype)  # 存储实际数据
        self.ptr = empty(N_part + 1, dtype=int)   # 存储每项数据的起始位置
        # 计算起始指针
        self.ptr[0] = 0
        for i in range(1, N_part + 1):
            self.ptr[i] = self.ptr[i - 1] + parts[i - 1]

    def SetData(self, iPart, idx, val):
        pos = self.ptr[iPart] + idx
        self.data[pos] = val

    def GetData(self, iPart, idx):
        pos = self.ptr[iPart] + idx
        return self.data[pos]

    def GetNumPart(self, iPart):
        return self.ptr[iPart + 1] - self.ptr[iPart]

    def GetTotalNum(self):
        return len(self.ptr) -1

    def AddAttr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MarkerData:
    """单个边界 marker 在某个分区上的局部数据。"""
    def __init__(self, name='', elem_to_node=None, elem_type=None, local_to_global_bnd_elem=None):
        self.name = name
        self.elem_to_node = elem_to_node
        self.elem_type = elem_type
        self.local_to_global_bnd_elem = local_to_global_bnd_elem

    def GetNumElem(self):
        if self.elem_to_node is None:
            return 0
        return self.elem_to_node.GetTotalNum()


class CommPattern:
    """点数据交换表，CSR 的 part 编号就是 MPI rank。"""
    def __init__(self, send_nodes=None, recv_nodes=None, send_ranks=None, recv_ranks=None,
                 send_point_global=None, recv_point_global=None):
        self.send_nodes = send_nodes
        self.recv_nodes = recv_nodes
        self.send_ranks = send_ranks
        self.recv_ranks = recv_ranks
        self.send_point_global = send_point_global
        self.recv_point_global = recv_point_global

    def GetNumSendRank(self):
        if self.send_ranks is None:
            return 0
        return len(self.send_ranks)

    def GetNumRecvRank(self):
        if self.recv_ranks is None:
            return 0
        return len(self.recv_ranks)


class PartitionMesh:
    """FVM 点图分区后的单个 rank 局部网格。"""
    def __init__(self, rank=0, size=1, n_dim=0):
        self.rank = rank
        self.size = size
        self.n_dim = n_dim

        self.n_point = 0
        self.n_point_domain = 0
        self.n_point_ghost = 0
        self.n_elem = 0
        self.n_halo_layer = 0

        self.coords = None
        self.point_color = None
        self.point_halo_layer = None

        self.local_to_global_point = None
        self.global_to_local_point = None

        self.elem_to_node = None
        self.elem_type = None
        self.local_to_global_elem = None

        self.markers = []
        self.comm = None

    def AddMarker(self, marker):
        self.markers.append(marker)

    def GetNumMarker(self):
        return len(self.markers)


if __name__ == "__main__":
    part = array([3, 2, 4], dtype=int)
    csr = CSR(part)
    csr.SetData(0, 0, 1.0)
    csr.SetData(0, 1, 2.0)
    csr.SetData(0, 2, 3.0)
    csr.SetData(1, 0, 4.0)
    csr.SetData(1, 1, 5.0)
    csr.SetData(2, 0, 6.0)
    csr.SetData(2, 1, 7.0)
    csr.SetData(2, 2, 8.0)
    csr.SetData(2, 3, 9.0)

    print("CSR Data:", csr.data)
    print("CSR Ptr:", csr.ptr)
    print("Get Data (0, 1):", csr.GetData(0, 1))
    print("Get Data (1, 0):", csr.GetData(1, 0))
    print("Get Data (2, 3):", csr.GetData(2, 3))
    print("Number of parts in part 0:", csr.GetNumPart(0))
    print("Number of parts in part 1:", csr.GetNumPart(1))
    print("Number of parts in part 2:", csr.GetNumPart(2))


    
    