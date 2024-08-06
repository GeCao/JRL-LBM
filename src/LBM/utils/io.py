import numpy as np
import torch
import math
import os
import cv2
from PIL import Image
from typing import List
import torch.nn.functional as F
import imageio


def tensor2numpy_2d_(img):
    # Normalization
    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min)

    img = img.detach().cpu()
    img = img.permute(0, 2, 3, 1) * 255
    return img[0].numpy()


def save_img(tensor_input, filename):
    if len(tensor_input.shape) == 4:
        np_img = tensor2numpy_2d_(tensor_input)
    else:
        raise RuntimeError("To save an image, the tensor shape should be 4 or 5")

    cv2.imwrite(filename, cv2.flip(np_img, 0))


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def export_cylinder(
    save_path: str, floor_center: np.ndarray, ceil_center: np.ndarray, radius: float
):
    floor_center = floor_center.reshape(3)
    ceil_center = ceil_center.reshape(3)
    dtype = floor_center.dtype
    n_circle = 40
    n_z = 5
    z_axis = (ceil_center - floor_center) / np.linalg.norm(ceil_center - floor_center)
    y_axis = np.array([1.0, 2.0, 5.0], dtype=dtype)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    dh = np.linalg.norm(ceil_center - floor_center) / (n_z - 1.0)

    verts = np.zeros((n_circle * n_z + 2, 3), dtype=dtype)
    for i in range(n_z):
        layer_center = floor_center + (i * dh) * z_axis
        for j in range(n_circle):
            phi = j / n_circle * 2.0 * math.pi
            verts[i * n_circle + j] = (
                radius * math.cos(phi) * x_axis
                + radius * math.sin(phi) * y_axis
                + layer_center
            )
    verts[n_circle * n_z] = floor_center
    verts[n_circle * n_z + 1] = ceil_center

    faces = []
    for j in range(n_circle):
        faces.append([n_circle * n_z, (j + 1) % n_circle, j])

    for j in range(n_circle):
        faces.append(
            [
                n_circle * n_z + 1,
                j + n_circle * (n_z - 1),
                (j + 1) % n_circle + n_circle * (n_z - 1),
            ]
        )

    for i in range(n_z - 1):
        i_pos = i + 1
        for j in range(n_circle):
            j_pos = (j + 1) % n_circle
            faces.append([i * n_circle + j, i * n_circle + j_pos, i_pos * n_circle + j])
            faces.append(
                [i * n_circle + j_pos, i_pos * n_circle + j_pos, i_pos * n_circle + j]
            )

    faces = np.array(faces, dtype=np.int32)

    export_asset(
        save_path=save_path,
        vertices=torch.from_numpy(verts),
        faces=torch.from_numpy(faces),
    )


def export_box(save_path: str, res: List[int], width: float = 0.01):
    dtype = np.float32
    AABB_box = np.array([[0, 0, 0], [res[0], res[1], res[2]]], dtype=dtype)
    AABB_box = (AABB_box - np.array([res], dtype=dtype) / 2.0) * (2.0 / max(res))

    xyz = np.array([AABB_box[0, 0], AABB_box[0, 1], AABB_box[0, 2]], dtype=dtype)
    Xyz = np.array([AABB_box[1, 0], AABB_box[0, 1], AABB_box[0, 2]], dtype=dtype)
    xYz = np.array([AABB_box[0, 0], AABB_box[1, 1], AABB_box[0, 2]], dtype=dtype)
    XYz = np.array([AABB_box[1, 0], AABB_box[1, 1], AABB_box[0, 2]], dtype=dtype)
    xyZ = np.array([AABB_box[0, 0], AABB_box[0, 1], AABB_box[1, 2]], dtype=dtype)
    XyZ = np.array([AABB_box[1, 0], AABB_box[0, 1], AABB_box[1, 2]], dtype=dtype)
    xYZ = np.array([AABB_box[0, 0], AABB_box[1, 1], AABB_box[1, 2]], dtype=dtype)
    XYZ = np.array([AABB_box[1, 0], AABB_box[1, 1], AABB_box[1, 2]], dtype=dtype)

    def add_box(xyz_, XYZ_, np_vertices_, np_faces_):
        dtype = xyz_.dtype
        eps = 0.01

        dist = np.linalg.norm(XYZ_ - xyz_)
        normal = (XYZ_ - xyz_) / dist
        tangent = np.array([0.0, 0.0, 0.0], dtype=dtype)
        bitangent = np.array([0.0, 0.0, 0.0], dtype=dtype)
        if normal.sum() < 0.0:
            xyz_, XYZ_ = XYZ_, xyz_
            normal = -normal
        if abs(normal[0]) >= eps:
            assert abs(normal[1]) < eps and abs(normal[2]) < eps
            tangent = np.array([0.0, 1.0, 0.0], dtype=dtype)
            bitangent = np.array([0.0, 0.0, 1.0], dtype=dtype)
        elif abs(normal[1]) >= eps:
            assert abs(normal[0]) < eps and abs(normal[2]) < eps
            tangent = np.array([0.0, 0.0, 1.0], dtype=dtype)
            bitangent = np.array([1.0, 0.0, 0.0], dtype=dtype)
        elif abs(normal[2]) >= eps:
            assert abs(normal[0]) < eps and abs(normal[1]) < eps
            tangent = np.array([1.0, 0.0, 0.0], dtype=dtype)
            bitangent = np.array([0.0, 1.0, 0.0], dtype=dtype)

        xyz_ = xyz_ - width * tangent - width * bitangent
        XYZ_ = XYZ_ + width * tangent + width * bitangent

        x_, y_, z_ = xyz_[0], xyz_[1], xyz_[2]
        X_, Y_, Z_ = XYZ_[0], XYZ_[1], XYZ_[2]
        new_vertices_ = np.array(
            [
                [x_, y_, z_],
                [X_, y_, z_],
                [X_, Y_, z_],
                [x_, Y_, z_],
                [x_, y_, Z_],
                [X_, y_, Z_],
                [X_, Y_, Z_],
                [x_, Y_, Z_],
            ],
            dtype=dtype,
        )
        new_faces_ = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 4],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        )
        vert_offset = np_vertices_.shape[-2]
        np_vertices_ = np.concatenate((np_vertices_, new_vertices_), axis=-2)
        np_faces_ = np.concatenate((np_faces_, new_faces_ + vert_offset), axis=-2)

        return [np_vertices_, np_faces_]

    np_vertices = np.zeros((0, 3), dtype=dtype)
    np_faces = np.zeros((0, 3), dtype=np.int32)
    # Add Cubes
    # Add Legs on z-plane
    np_vertices, np_faces = add_box(xyz, Xyz, np_vertices, np_faces)
    np_vertices, np_faces = add_box(Xyz, XYz, np_vertices, np_faces)
    np_vertices, np_faces = add_box(xYz, XYz, np_vertices, np_faces)
    np_vertices, np_faces = add_box(xyz, xYz, np_vertices, np_faces)

    np_vertices, np_faces = add_box(xyZ, XyZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(XyZ, XYZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(xYZ, XYZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(xyZ, xYZ, np_vertices, np_faces)

    np_vertices, np_faces = add_box(xyz, xyZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(Xyz, XyZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(xYz, xYZ, np_vertices, np_faces)
    np_vertices, np_faces = add_box(XYz, XYZ, np_vertices, np_faces)

    export_asset(
        save_path=save_path,
        vertices=torch.from_numpy(np_vertices),
        faces=torch.from_numpy(np_faces),
    )


def export_asset(save_path: str, vertices: torch.Tensor, faces: torch.Tensor):
    np_faces = faces.reshape(-1, 3).to(torch.int).cpu().numpy()
    np_vertices = vertices.reshape(-1, 3).cpu().numpy()
    if np_faces.min() == 0:
        np_faces = np_faces + 1
    with open(save_path, "w") as f:
        f.write("# OBJ file\n")
        for i in range(np_vertices.shape[0]):
            f.write(
                "v {} {} {}\n".format(
                    np_vertices[i, 0], np_vertices[i, 1], np_vertices[i, 2]
                )
            )
        for j in range(np_faces.shape[0]):
            f.write(
                "f {} {} {}\n".format(np_faces[j, 0], np_faces[j, 1], np_faces[j, 2])
            )
    f.close()


def dump_2d_plt_file_balance(filename, np_C, np_vel, np_h, B):
    C_shape = [*np_C.shape]
    fo = open(filename, "w")
    fo.write('TITLE ="Magnetic internal force"\n')
    fo.write(
        'VARIABLES = "X" "Y" "C" "U" "V" "F0" "F1" "F2" "F3" "F4" "F5" "F6" "F7" "F8"\n'
    )
    fo.write("ZONE I={}, J={}\n".format(C_shape[-1], C_shape[-2]))
    fo.write("F=POINT\n")
    for j in range(C_shape[-2]):
        for i in range(C_shape[-1]):
            fo.write(
                "{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                    i,
                    j,
                    np_C[B, 0, j, i],
                    np_vel[B, 1, j, i],
                    np_vel[B, 2, j, i],
                    np_h[B, 0, j, i],
                    np_h[B, 1, j, i],
                    np_h[B, 2, j, i],
                    np_h[B, 3, j, i],
                    np_h[B, 4, j, i],
                    np_h[B, 5, j, i],
                    np_h[B, 6, j, i],
                    np_h[B, 7, j, i],
                    np_h[B, 8, j, i],
                )
            )
    fo.close()


def read_2d_plt_file_balance(filename, np_C, np_vel, np_h, B):
    C_shape = [*np_C.shape]
    fo = open(filename, "r")
    fo.readline()
    fo.readline()
    fo.readline()
    fo.readline()
    for j in range(C_shape[-2]):
        for i in range(C_shape[-1]):
            this_line = fo.readline().split(" ")
            np_C[B, 0, j, i] = float(this_line[2])
            np_vel[B, 0, j, i] = float(this_line[3])
            np_vel[B, 1, j, i] = float(this_line[4])
            np_h[B, 0, j, i] = float(this_line[5])
            np_h[B, 1, j, i] = float(this_line[6])
            np_h[B, 2, j, i] = float(this_line[7])
            np_h[B, 3, j, i] = float(this_line[8])
            np_h[B, 4, j, i] = float(this_line[9])
            np_h[B, 5, j, i] = float(this_line[10])
            np_h[B, 6, j, i] = float(this_line[11])
            np_h[B, 7, j, i] = float(this_line[12])
            np_h[B, 8, j, i] = float(this_line[13])
    fo.close()

    return [np_C, np_vel, np_h]


def read_2d_plt_file_C_rho(filename):
    fo = open(filename, "r")
    line = fo.readline()
    line = fo.readline()
    line = fo.readline()
    I_loc1 = line.find("I=")
    I_loc2 = line.find(", J=")
    res_x = int(line[I_loc1 + 2 : I_loc2])
    res_y = int(line[I_loc2 + 4 : -1])
    line = fo.readline()

    np_C = np.zeros((1, 1, res_y, res_x), dtype=np.float32)
    np_density = np.zeros((1, 1, res_y, res_x), dtype=np.float32)
    np_u = np.zeros((1, 1, res_y, res_x), dtype=np.float32)
    np_v = np.zeros((1, 1, res_y, res_x), dtype=np.float32)
    for j in range(res_y):
        for i in range(res_x):
            this_line = fo.readline()[:-1].split(" ")
            np_C[0, 0, j, i] = float(this_line[2])
            np_density[0, 0, j, i] = float(this_line[3])
            np_u[0, 0, j, i] = float(this_line[4])
            np_v[0, 0, j, i] = float(this_line[5])
    fo.close()

    return [np_C, np_density, np_u, np_v]


def dump_2d_plt_file_C_rho(filename, np_C, np_density, np_u, np_v, B, C):
    density_shape = [*np_density.shape]
    fo = open(filename, "w")
    fo.write('TITLE ="Magnetic internal force"\n')
    fo.write('VARIABLES = "X" "Y" "C" "RHO" "U" "V" "NormX" "NormY"\n')
    fo.write("ZONE I={}, J={}\n".format(density_shape[-1], density_shape[-2]))
    fo.write("F=POINT\n")
    if np_u.shape[-1] == np_density.shape[-1] + 1:
        np_u = 0.5 * (np_u[..., 1:] + np_u[..., :-1])
    if np_v.shape[-2] == np_density.shape[-2] + 1:
        np_v = 0.5 * (np_v[..., 1:, :] + np_v[..., :-1, :])
    for j in range(density_shape[-2]):
        for i in range(density_shape[-1]):
            fo.write(
                "{} {} {} {} {} {} {} {}\n".format(
                    i,
                    j,
                    np_C[B, C, j, i],
                    np_density[B, C, j, i],
                    np_u[B, C, j, i],
                    np_v[B, C, j, i],
                    i / 12.5,
                    j / 12.5,
                )
            )
    fo.close()


def dump_2d_plt_file_single(filename, np_density, np_u, np_v, B, C):
    density_shape = [*np_density.shape]
    fo = open(filename, "w")
    fo.write('TITLE ="Magnetic internal force"\n')
    fo.write('VARIABLES = "X" "Y" "RHO" "U" "V" \n')
    fo.write("ZONE I={}, J={}\n".format(density_shape[-1], density_shape[-2]))
    fo.write("F=POINT\n")
    np_u = 0.5 * (np_u[..., 1:] + np_u[..., :-1])
    np_v = 0.5 * (np_v[..., 1:, :] + np_v[..., :-1, :])
    for j in range(density_shape[-2]):
        for i in range(density_shape[-1]):
            fo.write(
                "{} {} {} {} {}\n".format(
                    i, j, np_density[B, C, j, i], np_u[B, C, j, i], np_v[B, C, j, i]
                )
            )
    fo.close()


def dump_smoke_pbrt(filename: str, density: torch.Tensor, B: int = 0, C: int = 0):
    res = [*density.shape[-3:]]
    fo = open(filename, "w")
    fo.write('MakeNamedMedium "smoke"\n')
    fo.write(f'        "integer nx" [ {res[-1]} ] \n')
    fo.write(f'        "integer ny" [ {res[-2]} ] \n')
    fo.write(f'        "integer nz" [ {res[-3]} ] \n')
    scale = 1.0 / max(res)
    fo.write(
        f' "point p0" [ 0.0 0.0 0.0 ] "point p1" [{res[-1] * scale} {res[-2] * scale} {res[-3] * scale} ] \n'
    )
    fo.write('        "float density" [')
    np_density = density.cpu().numpy()
    for k in range(res[-3]):
        for j in range(res[-2]):
            for i in range(res[-1]):
                fo.write(" {:.5f}".format(np_density[B, C, k, j, i]))

    fo.write(' ]\n        "string type" [ "heterogeneous" ] \n')
    fo.close()
