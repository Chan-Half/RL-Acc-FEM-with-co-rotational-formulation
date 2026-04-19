"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_cuda_MatrixStruct.py
#  @brief:     fem cpu to cuda matrix
#  @author:    Hao Chen
#  @version:   1.0
#  @version:   1.0
#  @date:      2023.04.03
###########################################################################################
"""
import numpy as np
import pycuda.driver as cuda


class MatrixStruct(object):
    def __init__(self, array):
        self.cptr = None
        if array.dtype == np.float32 or array.dtype == np.int32:
            array = np.array(array, dtype=array.dtype)
            self.shape, self.dtype = array.shape, array.dtype
            if len(array.shape) == 1:
                self.width = np.int32(self.shape[0])
                self.height = np.int32(0)
                self.padding = np.int32(0)
            if len(array.shape) == 2:
                self.width = np.int32(self.shape[1])
                self.height = np.int32(self.shape[0])
                self.padding = np.int32(0)
            if len(self.shape) == 3:
                self.width = np.int32(self.shape[2])
                self.height = np.int32(self.shape[1])
                self.padding = np.int32(self.shape[0])
        if array.dtype == np.float64:
            array = np.array(array, dtype=array.dtype)
            self.shape, self.dtype = array.shape, array.dtype
            if len(array.shape) == 1:
                self.width = np.int32(self.shape[0])
                self.height = np.int32(0)
                self.padding = np.int32(0)
            if len(array.shape) == 2:
                self.width = np.int32(self.shape[1])
                self.height = np.int32(self.shape[0])
                self.padding = np.int32(0)
            if len(self.shape) == 3:
                self.width = np.int32(self.shape[2])
                self.height = np.int32(self.shape[1])
                self.padding = np.int32(self.shape[0])

        self.stride = self.width
        self.elements = cuda.to_device(array)  # 分配内存并拷贝数组数据至device，返回其地址

    def send_to_gpu(self):
        self.cptr = cuda.mem_alloc(self.nbytes())  # 分配一个C结构体所占的内存
        cuda.memcpy_htod(int(self.cptr), self.width.tobytes())  # 拷贝数据至device，下同
        cuda.memcpy_htod(int(self.cptr) + 4, self.height.tobytes())
        cuda.memcpy_htod(int(self.cptr) + 8, self.stride.tobytes())
        cuda.memcpy_htod(int(self.cptr) + 12, self.padding.tobytes())
        cuda.memcpy_htod(int(self.cptr) + 16, np.intp(int(self.elements)).tobytes())

    def d2h_row(self, c_k):
        return cuda.from_device(self.elements, (c_k, self.shape[1]), self.dtype)  # 从device取回数组数据

    def d2h(self):
        return cuda.from_device(self.elements, self.shape, self.dtype)  # 从device取回数组数据

    def nbytes(self):
        return self.width.nbytes * 4 + np.intp(0).nbytes

    def rehtod(self, arr):
        cuda.memcpy_htod(self.elements, np.array(arr, dtype=self.dtype))


    def rehtod2X(self, arr):
        """
        专门用于将二维 Numpy 数组重新拷贝到已分配的显存中
        """
        # 1. 转换为 Numpy 数组（使用 np.asarray 避免类型相同时产生不必要的拷贝）
        arr_np = np.asarray(arr, dtype=self.dtype)

        # 2. 确保传入的是二维数组
        if len(arr_np.shape) != 2:
            raise ValueError(f"rehtod2X 期望传入二维数组，实际传入了 {len(arr_np.shape)} 维数组。")

        # 3. 校验形状，防止显存越界写坏上下文
        # 注意：这里对比的是初始化时保存的 self.height 和 self.width
        if int(self.height) != arr_np.shape[0] or int(self.width) != arr_np.shape[1]:
            raise ValueError(f"形状不匹配：当前显存分配的矩阵形状为 ({self.height}, {self.width})，"
                             f"而传入的数组形状为 {arr_np.shape}。如需更改大小，请重新实例化。")

        # 4. 关键修复：确保数组在内存中是连续的 (C-contiguous)
        if not arr_np.flags['C_CONTIGUOUS']:
            arr_np = np.ascontiguousarray(arr_np)

        # 5. 安全拷贝：使用 .ravel() 把它当作一维线性 buffer 传入
        # 因为我们已经确保了内存连续，.ravel() 不会产生新拷贝，只是改变视图的维度，对底层 C 函数非常友好
        cuda.memcpy_htod(self.elements, arr_np.ravel())


