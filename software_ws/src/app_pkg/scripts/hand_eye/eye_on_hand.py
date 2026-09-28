#!/usr/bin/env python
# coding: utf-8
import cv2
import transforms3d as tfs
import numpy as np
import math
import os
import sys
#插入一个新的路径，方便导入自己的模块
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/realsense_control_pkg/scripts")
import util
import numpy as np



def get_matrix_eular_radu_hand(x,y,z,rx,ry,rz):
    #rmat = tfs.euler.euler2mat(math.radians(rx),math.radians(ry),math.radians(rz))
    rmat = tfs.euler.euler2mat(rx, ry, rz)
    #rmat = rmat.T
    p=np.asarray((x,y,z))
    #p=-np.dot(rmat, p)
    rmat = tfs.affines.compose(np.squeeze(p), rmat, [1, 1, 1])
    return rmat


def get_matrix_eular_radu(x,y,z,rx,ry,rz):
    #rmat = tfs.euler.euler2mat(math.radians(rx),math.radians(ry),math.radians(rz))
    rmat = tfs.euler.euler2mat(rx, -ry, rz)
    #rmat = rmat.T
    p=np.asarray((x,y,z))
    #p=-np.dot(rmat, p)
    rmat = tfs.affines.compose(np.squeeze(p), rmat, [1, 1, 1])
    return rmat

def get_rv_R(rx,ry,rz):
    # rmat = tfs.euler.euler2mat(rx, ry, rz)
    rmat = util.rv2rm(rx, ry, rz)
    #rmat = rmat.T
    return rmat

def get_r(rx,ry,rz):
    rmat = tfs.euler.euler2mat(rx, ry, rz)
    # rmat = util.rv2rm(rx, ry, rz)
    #rmat = rmat.T
    return rmat

def get_t(x,y,z):
    t=np.asarray((x,y,z))
    #t= -np.dot(R,t)
    return t

def skew(v):
    return np.array([[0,-v[2],v[1]],
                     [v[2],0,-v[0]],
                     [-v[1],v[0],0]])

def rot2quat_minimal(m):
    quat =  tfs.quaternions.mat2quat(m[0:3,0:3])
    return quat[1:]

def quatMinimal2rot(q):
    p = np.dot(q.T,q)
    w = np.sqrt(np.subtract(1,p[0][0]))
    return tfs.quaternions.quat2mat([w,q[0],q[1],q[2]])

# hand是标定码相对相机坐标系下的位姿，前三个数据为位置数据，单位为m;
# 后三个数据为位姿数据，欧拉角，单位为弧度制




# camera = [0.0249, 0.04587, 0.34642, 3.05596, 0.08794, -1.41908,
#         -0.04621, 0.0286, 0.30052, 2.81512, 0.12543, -1.44512,
#         0.12977, 0.0254, 0.27028, -3.02733, -0.05013, -1.42268,
#         0.06288, 0.00511, 0.19569, 3.08298, -0.00649, -1.42419,
#         0.02252, 0.00732, 0.41192, 3.11198, -0.26681, -1.43701,
#         0.16688, 0.05356, 0.32579, -2.9053, -0.25003, -1.48997,
#         0.03484, 0.01413, 0.27761, -3.06764, 0.24723, -1.41979,
#         0.05269, -0.02528, 0.25608, 2.7575, 0.14793, -2.79279,
#         0.03561, -0.01106, 0.26496, -2.89112, -0.09349, -2.46176,
#         0.07574, 0.01355, 0.24687, 3.05446, 0.21193, -0.4331]

# hand =[-0.35461, 0.05923, 0.36496, -3.13114, 0.10378, 2.54603,
#         -0.3562, 0.20374, 0.27913, -2.94436, 0.2851, 2.5647,
#         -0.26946, -0.09688, 0.27835, 3.08697, -0.12032, 2.5463,
#         -0.27491, 0.01333, 0.22126, -3.09439, 0.00888, 2.5363,
#         -0.17543, 0.04608, 0.3916, -2.97527, -0.21126, 2.50399,
#         -0.22831, -0.17939, 0.28709, 3.12276, -0.3505, 2.50238,
#         -0.36476, 0.00335, 0.3075, 2.9416, 0.14202, 2.51698,
#         -0.37837, 0.02364, 0.29954, 2.86936, 0.28816, 1.15972,
#         -0.27444, -0.12607, 0.25364, -3.02659, -0.25089, 1.49672,
#         -0.35205, 0.09209, 0.27317, -2.97656, 0.14895, -2.73467]


camera = [0.01413, 0.00578, 0.45371, 3.05958, -0.0435, -2.80918,
          0.03486, -0.00719, 0.42397, -3.07897, -0.07087, -2.93271,
          0.05476, -0.02353, 0.4327, -2.9883, -0.16002, -2.94629,
          0.03897, 0.00651, 0.42216, -3.11931, -0.03165, -2.91786,
          0.05683, 0.00181, 0.41292, 3.05904, 0.20372, -2.91802,
          0.10614, -0.04469, 0.4601, 2.96903, 0.11409, -2.74095,
          0.06614, -0.02288, 0.46938, 2.98905, 0.02826, -2.62258,
          0.07367, -0.01022, 0.47746, -3.08841, -0.23548, -2.92427,
          0.02627, -0.01461, 0.47655, -3.13623, -0.02939, -2.92497,
          0.0569, -0.041, 0.46251, 3.03318, 0.1009, -2.92413]

hand =[-0.2954, 0.10088, 0.32876, -3.10957, 0.062, 3.10977,
       -0.24311, 0.07852, 0.29449, -3.0863, -0.02428, 2.98718,
       -0.1858, 0.08886, 0.2851, -2.9821, -0.12696, 2.9763,
       -0.26119, 0.05434, 0.29495, 3.13437, -0.01661, 3.0034,
       -0.27458, -0.0206, 0.28613, 2.92713, 0.02958, 3.00866,
       -0.24434, -0.01646, 0.35028, 3.01718, 0.13996, -3.10216,
       -0.28381, 0.0564, 0.35604, 3.12337, 0.14084, -2.98062,
       -0.23055, 0.10866, 0.344, -2.91241, 0.00069, 3.00073,
       -0.24583, 0.07536, 0.3488, -3.137, -0.00609, 2.99727,
       -0.24225, -0.00267, 0.3419, 2.98213, 0.05203, 2.99692]






def e_h_H(hand,camera):
    Hgs,Hcs = [],[]
    for i in range(0,len(hand),6):
    #for i in range(0, 10*6, 6):
        Hgs.append(get_matrix_eular_radu_hand(hand[i],hand[i+1],hand[i+2],hand[i+3],hand[i+4],hand[i+5]))
        Hcs.append(get_matrix_eular_radu(camera[i],camera[i+1],camera[i+2],camera[i+3],camera[i+4],camera[i+5]))

    Hgijs = []
    Hcijs = []
    A = []
    B = []
    size = 0
    for i in range(len(Hgs)):
        for j in range(i+1,len(Hgs)):
            size += 1
            Hgij = np.dot(np.linalg.inv(Hgs[j]),Hgs[i])
            Hgijs.append(Hgij)
            Pgij = np.dot(2,rot2quat_minimal(Hgij))
            
            Hcij = np.dot(Hcs[j],np.linalg.inv(Hcs[i]))
            Hcijs.append(Hcij)
            Pcij = np.dot(2,rot2quat_minimal(Hcij))
            
            A.append(skew(np.add(Pgij,Pcij)))
            B.append(np.subtract(Pcij,Pgij))
    MA = np.asarray(A).reshape(size*3,3)
    MB = np.asarray(B).reshape(size*3,1)
    Pcg_  =  np.dot(np.linalg.pinv(MA),MB)
    pcg_norm = np.dot(np.conjugate(Pcg_).T,Pcg_)
    Pcg = np.sqrt(np.add(1,np.dot(Pcg_.T,Pcg_)))
    Pcg = np.dot(np.dot(2,Pcg_),np.linalg.inv(Pcg))
    Rcg = quatMinimal2rot(np.divide(Pcg,2)).reshape(3,3)


    A = []
    B = []
    id = 0
    for i in range(len(Hgs)):
        for j in range(i+1,len(Hgs)):
            Hgij = Hgijs[id]
            Hcij = Hcijs[id]
            A.append(np.subtract(Hgij[0:3,0:3],np.eye(3,3)))
            B.append(np.subtract(np.dot(Rcg,Hcij[0:3,3:4]),Hgij[0:3,3:4]))
            id += 1

    MA = np.asarray(A).reshape(size*3,3)
    MB = np.asarray(B).reshape(size*3,1)
    Tcg = np.dot(np.linalg.pinv(MA),MB).reshape(3,)
    print(tfs.affines.compose(Tcg,np.squeeze(Rcg),[1,1,1]))
    return tfs.affines.compose(Tcg,np.squeeze(Rcg),[1,1,1])

def eye_in_hand(hand,camera):
    R_all_end_to_base_1, T_all_end_to_base_1   = [], []
    R_all_chess_to_cam_1, T_all_chess_to_cam_1 = [], []
    for i in range(0, len(hand), 6):
        r = get_r(hand[i + 3], hand[i + 4], hand[i + 5])
        # r_1 =np.linalg.inv(r)
        t = get_t(hand[i], hand[i + 1], hand[i + 2])
        # t_1 = -np.dot(r_1, t)
        T_all_end_to_base_1.append(t)
        R_all_end_to_base_1.append(r)



        r = get_r(camera[i + 3], camera[i + 4], camera[i + 5])
        # r = np.linalg.inv(r)
        t = get_t(camera[i], camera[i + 1], camera[i + 2])
        # t = -np.dot(r, t)
        T_all_chess_to_cam_1.append(t)
        R_all_chess_to_cam_1.append(r)

    R, T = cv2.calibrateHandEye(R_all_end_to_base_1, T_all_end_to_base_1, R_all_chess_to_cam_1, T_all_chess_to_cam_1)

    np.set_printoptions(suppress=True, precision=10)
    print('R=', R)
    print('T=', T)

if __name__ == "__main__":
    eye_in_hand(hand,camera)
    #print('  ')
    #e_h_H(hand, camera)

