'''
Filename: mpifvm2d-py/src/run_partition.py
Created Date: Sunday, May 17th 2026
Author: Lixiang Jiang

All Copyright (c) 2026 Lixiang Jiang. All rights reserved.
'''

import argparse
import os

import numpy as np
from mpi4py import MPI


from mesh import read_SU2_mesh
from partition import (build_adjacency, graph_partition, build_partition_meshes,
                       build_comm_patterns, check_partition_meshes, parts_info)
from tools import write_tecplot


def parse_args():
    parser = argparse.ArgumentParser(description='Build FVM point partitions and two-layer halo data structures.')
    parser.add_argument('--mesh', default='mesh/mesh_RAE2822_turb.su2', help='Path to SU2 mesh file.')
    parser.add_argument('--halo-layers', type=int, default=2, help='Number of FVM halo layers.')
    parser.add_argument('--output', default='output/mesh_partition.plt', help='Tecplot file for global point partition.')
    parser.add_argument('--no-output', action='store_true', help='Do not write Tecplot output.')
    return parser.parse_args()


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_args()

    if rank == 0:
        mesh_data = read_SU2_mesh(args.mesh)
        adjacency = build_adjacency(mesh_data)
        point_color = graph_partition(adjacency, n_partitions=size)
        part_meshes = build_partition_meshes(mesh_data, point_color, size, n_halo_layer=args.halo_layers)
        build_comm_patterns(part_meshes, point_color)
        check_partition_meshes(part_meshes, point_color)

        print('Point partition counts:', parts_info(point_color))
        for part_mesh in part_meshes:
            print('Rank {}: n_point={}, n_domain={}, n_ghost={}, n_elem={}, send_ranks={}, recv_ranks={}'.format(
                part_mesh.rank, part_mesh.n_point, part_mesh.n_point_domain, part_mesh.n_point_ghost,
                part_mesh.n_elem, list(part_mesh.comm.send_ranks), list(part_mesh.comm.recv_ranks)))

        if not args.no_output:
            nodes, elements, boundaries = mesh_data
            plot_nodes = np.concatenate((nodes, point_color.astype(np.float64).reshape(-1, 1)), axis=1)
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            write_tecplot(plot_nodes, elements, ['X', 'Y', 'Parts'], args.output)
    else:
        part_meshes = None

    local_mesh = comm.scatter(part_meshes, root=0)
    print('MPI rank {} owns partition {} with {} domain points, {} ghost points, {} elements.'.format(
        rank, local_mesh.rank, local_mesh.n_point_domain, local_mesh.n_point_ghost, local_mesh.n_elem), flush=True)


if __name__ == '__main__':
# mpiexec -np 5 python3 src/run_partition.py --mesh mesh/mesh_RAE2822_turb.su2 --halo-layers 2
    main()
