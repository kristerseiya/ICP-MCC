
import numpy as np
import open3d as o3d
import icp
import basic
import copy
import argparse
import matplotlib.pyplot as plt

def mesh2pcd(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    return pcd

def generate_random_transformation(max_deg=5., max_dist=20.):
    theta_x = np.random.uniform() * max_deg / 360. * 2 * np.pi
    theta_y = np.random.uniform() * max_deg / 360. * 2 * np.pi
    theta_z = np.random.uniform() * max_deg / 360. * 2 * np.pi
    translate = np.random.uniform(size=(3,)) * max_dist
    rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    r = np.matmul(rx, ry)
    r = np.matmul(r, rz)
    Rt = np.eye(4)
    Rt[:3, :3] = r
    Rt[:3, 3] = translate
    return Rt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='path to a point cloud', required=True)
    parser.add_argument('--input2', '-i2', help='path to a second point cloud', default=None)
    parser.add_argument('--metric', '-m', help='\"mse\" for normal icp, \"mcc\" '
                                               'for maximum correntropy criterion', type=str, default='mse')
    parser.add_argument('--iter', help='number of iteration in ICP', type=int, default=20)
    parser.add_argument('--bi_dir', '-b', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--transform', '-t', type=str, default=None)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()


    ext = args.input.split('.')[-1]
    if ext == 'ply':
        pcd1 = o3d.io.read_point_cloud(args.input)
    elif ext == 'stl':
        mesh1 = o3d.io.read_triangle_mesh(args.input)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = mesh1.vertices
    else:
        raise ValueError('Extension Not Supported')

    pcd1 = pcd1.voxel_down_sample(1.0)
    points1 = np.array(pcd1.points)


    if args.input2 != None:
        ext = args.input2.split('.')[-1]
        if ext == 'ply':
            pcd1 = o3d.io.read_point_cloud(args.input2)
        elif ext == 'stl':
            mesh1 = o3d.io.read_triangle_mesh(args.input2)
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = mesh1.vertices
        else:
            raise ValueError('Extension Not Supported')

        pcd2 = pcd2.voxel_down_sample(1.0)
        points2 = np.array(pcd2.points)
    else:
        if args.transform == None:
            T = generate_random_transformation()
        else:
            f = open(args.transform, 'r')
            T = list()
            for line in f:
                if line.strip() != '':
                    row = [float(x) for x in line.split()]
                    T.append(row)
            T = np.array(T)
            assert(T.shape[0] == 4)
            assert(T.shape[1] == 4)

        print('Target Transformation')
        for row in T:
            print('{:f} {:f} {:f} {:f}'.format(row[0], row[1], row[2], row[3]))
        print()

        pcd2 = o3d.geometry.PointCloud()
        points2 = basic.transform(points1, T)
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2 = pcd2.voxel_down_sample(0.3)
        points2 = np.array(pcd2.points)

    pcd1.paint_uniform_color([97./255, 208./255, 255./255])
    pcd2.paint_uniform_color([217./255, 146./255, 1./255])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()

    if args.monitor:
        T = icp.icp(points1, points2, 2, mode=args.metric.upper(),
                    iter=args.iter,
                    bidirectional=args.bi_dir,
                    verbose=args.verbose, real_T=T)
    else:
        T = icp.icp(points1, points2, 2, mode=args.metric.upper(),
                    iter=args.iter,
                    bidirectional=args.bi_dir,
                    verbose=args.verbose)


    print('Result Transformation')
    for row in T:
        print('{:f} {:f} {:f} {:f}'.format(row[0], row[1], row[2], row[3]))
    new_points1 = basic.transform(points1, T)

    pcd1.points = o3d.utility.Vector3dVector(new_points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd1.paint_uniform_color([97./255, 208./255, 255./255])
    pcd2.paint_uniform_color([217./255, 146./255, 1./255])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.run()
    vis.destroy_window()
