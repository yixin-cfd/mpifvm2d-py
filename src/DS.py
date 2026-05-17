'''
Filename: /home/yixin/Data/Code/python/mpifvm2d-py/src/DS.py
Created Date: Friday, May 15th 2026, 7:57:14 pm
Author: Lixiang Jiang 

All rights reserved.

Try to use the Fortran syntax style.!!!
'''
from mpi4py import MPI
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


    
    