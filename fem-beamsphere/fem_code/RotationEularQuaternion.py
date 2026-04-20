"""
###########################################################################################
#  @filename:  RotationEularQuaternion.py
#  @brief:     fem quaternion, Eular and rotation transform function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2026.04.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
import numpy as np
import math
# 旋转矩阵转四元数需要pyquaternion包
import time
from pyquaternion import Quaternion
# 四元数转旋转矩阵需要scipy
from scipy.spatial.transform import Rotation as Rot
import scipy.linalg as linalg


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# rotationMatrixToEulerAngles 用于旋转矩阵转欧拉角
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# eulerAnglesToRotationMatrix欧拉角转旋转矩阵
def eulerAnglesToRotationMatrix(theta, order):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    # R_x = linalg.expm(np.cross(np.eye(3), [1., 0., 0.] / linalg.norm([1, 0, 0]) * theta[0]))
    # R_y = linalg.expm(np.cross(np.eye(3), [0., 1., 0.] / linalg.norm([0, 1, 0]) * theta[1]))
    # R_z = linalg.expm(np.cross(np.eye(3), [0., 0., 1.] / linalg.norm([0, 0, 1]) * theta[2]))
    # R = np.dot(R_z, np.dot(R_y, R_x))
    if order == 'zyx':   # 先旋转z，再y，后x
        R = np.dot(R_x, np.dot(R_y, R_z))
    elif order == 'xyz' or order is None:   # 先x，再y，再z
        R = np.dot(R_z, np.dot(R_y, R_x))
    else:
        print('The Eular to Rotation have no order')
        R = np.eye(3)
    return R


# 旋转矩阵转四元数
def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    return q


# 四元数转轴角[x,y,z],单位向量代表方向，模长代表角度
def QuaternionToAxialAngle(q):
    AxialAngle = np.zeros(3, dtype=np.float32)
    half_theta = math.acos(q.w)
    sin_half_theta = math.sin(half_theta)
    if sin_half_theta != 0:
        AxialAngle[0] = q.x / sin_half_theta * half_theta * 2
        AxialAngle[1] = q.y / sin_half_theta * half_theta * 2
        AxialAngle[2] = q.z / sin_half_theta * half_theta * 2
    return AxialAngle


if __name__ == '__main__':
    # 初始化旋转矩阵
    rotationMat = np.array([[-0.90748313, -0.30075654, -0.29329146],
                            [-0.05041803, -0.61514386, 0.78680115],
                            [-0.41705203, 0.72879595, 0.54306912]])

    EulerAngles = rotationMatrixToEulerAngles(rotationMat)
    print("\nOutput Euler angles :\n{0}".format(EulerAngles))

    # 欧拉角转旋转矩阵
    rotationMat_1 = eulerAnglesToRotationMatrix(EulerAngles, 'xyz')
    print("\nR1 :\n{0}".format(rotationMat_1))


    # 旋转矩阵转四元数
    Quat = rotateToQuaternion(rotationMat)
    print("四元数x为: ", Quat.x, "\n四元数y为: ", Quat.y, "\n四元数z为: ", Quat.z, "\n四元数w为: ", Quat.w)

    # 四元数转旋转矩阵
    Rq = [Quat.x.astype(float), Quat.y.astype(float), Quat.z.astype(float),
          Quat.w.astype(float)]
    Rm = Rot.from_quat(Rq)
    rotation_matrix = Rm.as_matrix()
    print('rotation:\n', rotation_matrix)

    AxialAngle = QuaternionToAxialAngle(Quat)
    print(AxialAngle)
    print(linalg.expm(np.cross(np.eye(3), AxialAngle)))
