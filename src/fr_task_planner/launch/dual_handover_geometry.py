
#!/usr/bin/env python3

"""
DUAL-4A: 双臂交接几何计算。

只计算：
    T_world_object
    T_world_tcpA
    T_world_tcpB
    T_world_preB
    T_tcpB_object

不启动 ROS。
不连接机械臂。
不发送任何执行命令。
"""

import math
import os
import sys

import numpy as np
import yaml


def make_transform(rotation, translation):
    """由旋转矩阵和平移构造 4x4 齐次变换。"""

    T = np.eye(4)

    T[:3, :3] = rotation
    T[:3, 3] = translation

    return T


def inverse_transform(T):
    """计算齐次变换的逆。"""

    R = T[:3, :3]
    p = T[:3, 3]

    result = np.eye(4)

    result[:3, :3] = R.T
    result[:3, 3] = -R.T @ p

    return result


def rotation_to_quaternion(R):
    """将旋转矩阵转换为 ROS xyzw 四元数。"""

    trace = float(np.trace(R))

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0

        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    else:
        i = int(np.argmax(np.diag(R)))

        if i == 0:
            s = math.sqrt(
                1.0 + R[0, 0]
                - R[1, 1]
                - R[2, 2]
            ) * 2.0

            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s

        elif i == 1:
            s = math.sqrt(
                1.0 + R[1, 1]
                - R[0, 0]
                - R[2, 2]
            ) * 2.0

            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s

        else:
            s = math.sqrt(
                1.0 + R[2, 2]
                - R[0, 0]
                - R[1, 1]
            ) * 2.0

            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w])

    return q / np.linalg.norm(q)


def print_pose(name, T):

    p = T[:3, 3]

    q = rotation_to_quaternion(
        T[:3, :3]
    )

    print(f"\n{name}")

    print(
        "  xyz  =",
        np.round(p, 6).tolist()
    )

    print(
        "  xyzw =",
        np.round(q, 6).tolist()
    )


def check_rotation(name, R):

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-9
    ):
        raise RuntimeError(
            f"{name}: rotation is not orthogonal"
        )

    if not np.isclose(
        np.linalg.det(R),
        1.0,
        atol=1e-9
    ):
        raise RuntimeError(
            f"{name}: invalid rotation determinant"
        )


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 dual_handover_geometry.py "
            "<dual_handover.yaml>"
        )

    with open(
        os.path.expanduser(sys.argv[1]),
        "r",
        encoding="utf-8"
    ) as f:

        cfg = yaml.safe_load(f)

    if cfg["execution"]["enabled"]:
        raise RuntimeError(
            "DUAL-4A must remain geometry-only"
        )

    center_cfg = cfg["object"]["center_world"]

    object_center = np.array([
        center_cfg["x"],
        center_cfg["y"],
        center_cfg["z"]
    ], dtype=float)

    b_offset = float(
        cfg["handover"][
            "arm_b_grasp_offset_x_m"
        ]
    )

    pre_distance = float(
        cfg["handover"][
            "prehandover_b_distance_m"
        ]
    )

    length = float(
        cfg["object"]["length_m"]
    )

    if not (
        -length / 2.0 < b_offset < 0.0
    ):
        raise RuntimeError(
            "Arm B offset must be inside "
            "the negative-X half of the object"
        )

    if pre_distance <= 0:
        raise RuntimeError(
            "PreHandover distance must be positive"
        )

    # ----------------------------------------
    # 1. Object orientation
    #
    # object +Z -> world +X
    # object +X -> world +Y
    # object +Y -> world +Z
    #
    # This also preserves Arm A's existing
    # T_tcpA_object = Rx(pi).
    # ----------------------------------------

    R_world_object = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    check_rotation(
        "R_world_object",
        R_world_object
    )

    T_world_object = make_transform(
        R_world_object,
        object_center
    )

    # ----------------------------------------
    # 2. Existing Arm A grasp relationship
    #
    # T_tcpA_object:
    # translation = 0
    # rotation    = Rx(pi)
    # ----------------------------------------

    R_tcpA_object = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])

    T_tcpA_object = make_transform(
        R_tcpA_object,
        [0.0, 0.0, 0.0]
    )

    T_world_tcpA = (
        T_world_object
        @ inverse_transform(T_tcpA_object)
    )

    # ----------------------------------------
    # 3. Arm B orientation
    #
    # TCP +Z -> world +X
    # TCP +X -> world +Z
    # TCP +Y -> world -Y
    #
    # Finger plane: world XZ.
    # ----------------------------------------

    R_world_tcpB = np.array([
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])

    check_rotation(
        "R_world_tcpB",
        R_world_tcpB
    )

    b_position = object_center.copy()

    b_position[0] += b_offset

    T_world_tcpB = make_transform(
        R_world_tcpB,
        b_position
    )

    # ----------------------------------------
    # 4. PreHandover B:
    # world -X, 20 cm from Handover B.
    # Orientation unchanged.
    # ----------------------------------------

    T_world_preB = T_world_tcpB.copy()

    T_world_preB[0, 3] -= pre_distance

    # ----------------------------------------
    # 5. Object expressed in Arm B TCP.
    # ----------------------------------------

    T_tcpB_object = (
        inverse_transform(T_world_tcpB)
        @ T_world_object
    )

    # ----------------------------------------
    # 6. Verification
    # ----------------------------------------

    recovered_from_a = (
        T_world_tcpA
        @ T_tcpA_object
    )

    recovered_from_b = (
        T_world_tcpB
        @ T_tcpB_object
    )

    assert np.allclose(
        recovered_from_a,
        T_world_object,
        atol=1e-9
    )

    assert np.allclose(
        recovered_from_b,
        T_world_object,
        atol=1e-9
    )

    assert np.allclose(
        T_world_tcpA[:3, :3]
        @ np.array([0.0, 0.0, 1.0]),
        [-1.0, 0.0, 0.0]
    )

    assert np.allclose(
        T_world_tcpB[:3, :3]
        @ np.array([0.0, 0.0, 1.0]),
        [1.0, 0.0, 0.0]
    )

    assert np.allclose(
        T_world_preB[:3, 3]
        - T_world_tcpB[:3, 3],
        [-pre_distance, 0.0, 0.0]
    )

    # ----------------------------------------
    # 7. Print results
    # ----------------------------------------

    print("\n========== DUAL-4A ==========")

    print_pose(
        "OBJECT WORLD",
        T_world_object
    )

    print_pose(
        "ARM A HANDOVER TCP",
        T_world_tcpA
    )

    print_pose(
        "ARM B HANDOVER TCP",
        T_world_tcpB
    )

    print_pose(
        "ARM B PREHANDOVER TCP",
        T_world_preB
    )

    print_pose(
        "OBJECT RELATIVE TO ARM B TCP",
        T_tcpB_object
    )

    print(
        "\nDUAL-4A GEOMETRY PASS"
    )

    print(
        "NOTE: IK, collision and grasp "
        "feasibility are NOT yet verified."
    )


if __name__ == "__main__":
    main()