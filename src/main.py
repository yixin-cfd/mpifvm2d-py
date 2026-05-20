'''
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
                       build_comm_patterns, check_partition_meshes, parts_info, build_comm_pattern_parallel)
from tools import write_tecplot, visualize_mesh
from fvm import build_fvm_struct, Metrics, residual


def parse_args():
    parser = argparse.ArgumentParser(description='Build FVM point partitions and two-layer halo data structures.')
    parser.add_argument('--mesh', default='mesh/mesh_RAE2822_turb.su2', help='Path to SU2 mesh file.')
    parser.add_argument('--halo-layers', type=int, default=2, help='Number of FVM halo layers.')
    parser.add_argument('--output', default='output/mesh_partition.plt', help='Tecplot file for global point partition.')
    parser.add_argument('--no-output', action='store_true', help='Do not write Tecplot output.')
    parser.add_argument('--RCM_show', action='store_true', help='show RCM ordering.')
    parser.add_argument('--mesh_show', action='store_true', help='show mesh structure.')
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

        part_meshes = build_partition_meshes(
            mesh_data,
            point_color,
            size,
            n_halo_layer=args.halo_layers
        )

        print('Point partition counts:', parts_info(point_color))

        if not args.no_output:
            nodes, elements, boundaries = mesh_data
            plot_nodes = np.concatenate(
                (nodes, point_color.astype(np.float64).reshape(-1, 1)),
                axis=1
            )

            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            write_tecplot(plot_nodes, elements, ['X', 'Y', 'Parts'], args.output)

    else:
        point_color = None
        part_meshes = None

    # 每个 rank 需要知道 point_color，才能判断 ghost 点 owner
    point_color = comm.bcast(point_color, root=0)

    # rank0 构造好的 part_meshes 正常 scatter
    local_mesh = comm.scatter(part_meshes, root=0)

    # 并行构造通信表
    local_mesh = build_comm_pattern_parallel(local_mesh, point_color, comm)

    print(
        'MPI rank {} owns partition {} with {} domain points, {} ghost points, {} elements, send_ranks={}, recv_ranks={}.'.format(
            rank,
            local_mesh.rank,
            local_mesh.n_point_domain,
            local_mesh.n_point_ghost,
            local_mesh.n_elem,
            list(local_mesh.comm.send_ranks),
            list(local_mesh.comm.recv_ranks)
        ),
        flush=True
    )


    edges, vertices = build_fvm_struct(args, local_mesh, RCM_ordering=True)


    if args.mesh_show:
        visualize_mesh(local_mesh)

if __name__ == '__main__':
# mpiexec -np 5 python3 src/main.py --mesh mesh/mesh_RAE2822_turb.su2 --halo-layers 2
    main()
