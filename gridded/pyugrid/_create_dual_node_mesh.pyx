"""Utility function for creating the dual mesh for the nodes."""
from cython.view cimport array as cvarray
import numpy as np


def get_face_node_connectivity(
    const long[:, :] dual_edge_face_node_connectivity,
    const long[:, :] node_edge_connectivity,
    long n_dual_node,
    long nmax_face,
):
    """Create the face_node_connectivity for the dual node mesh.

    This function loops through every node and creates the corresponding
    elements.
    """
    cdef n_node = len(node_edge_connectivity)
    cdef n_edge = len(dual_edge_face_node_connectivity)

    cdef long[:, :] ret = cvarray(
        shape=(n_node, nmax_face), itemsize=sizeof(long), format="l"
    )

    cdef long n_dual_node_max = n_dual_node + n_edge
    cdef const long[:] edges

    cdef long i, j, node, i_edge
    cdef long[:] edge
    cdef long nmax_edge = node_edge_connectivity.shape[1]
    cdef long[:, :] dual_cells = cvarray(
        shape=(nmax_edge, 4), itemsize=sizeof(long), format="l"
    )
    cdef long[:, :] sorted_edges = cvarray(
        shape=(nmax_edge + 2, 2), itemsize=sizeof(long), format="l"
    )
    cdef long[:, :] dual_edges = cvarray(
        shape=(nmax_edge + 2, 2), itemsize=sizeof(long), format="l"
    )
    cdef long[:] dual_cell = cvarray(
        shape=(4, ), itemsize=sizeof(long), format="l"
    )

    cdef long n_cell_edges = nmax_edge
    cdef long start, end

    ret[:, :] = n_dual_node_max

    for node in range(n_node):
        edges = node_edge_connectivity[node]

        # clear the arrays
        dual_cells[:, :] = n_dual_node_max
        dual_edges[:, :] = n_dual_node_max
        sorted_edges[:, :] = n_dual_node_max
        if edges[0] == n_edge:
            continue

        for i in range(nmax_edge):
            i_edge = edges[i]
            if i_edge < n_edge:
                dual_cell[:] = dual_edge_face_node_connectivity[i_edge]
                n_cell_edges = i + 1

                # make sure that the node is in the first position
                if dual_cell[0] != node:
                    dual_cell[:2] = dual_cell[2:4]
                    dual_cell[2:4] = dual_edge_face_node_connectivity[i_edge, :2]

                dual_cells[i, :] = dual_cell

        dual_edges[:nmax_edge] = dual_cells[:, 1::2]

        # now check for missing nodes (indicates that the node is at the
        # edge of the mesh)
        for i in range(n_cell_edges):
            i_edge = edges[i]
            edge = dual_edges[i]
            if edge[0] == n_dual_node_max:
                edge[0] = n_dual_node + i_edge
                dual_edges[n_cell_edges, 0] = node
                dual_edges[n_cell_edges, 1] = n_dual_node + i_edge
                n_cell_edges = n_cell_edges + 1
            elif edge[1] == n_dual_node_max:
                edge[1] = n_dual_node + i_edge
                dual_edges[n_cell_edges, 1] = node
                dual_edges[n_cell_edges, 0] = n_dual_node + i_edge
                n_cell_edges = n_cell_edges + 1

        # now sort the edges
        sorted_edges[0, :] = dual_edges[0]
        for i in range(1, n_cell_edges):
            end = sorted_edges[i - 1, 1]
            for j in range(1, n_cell_edges):
                start = dual_edges[j, 0]
                if start == end:
                    sorted_edges[i] = dual_edges[j]
                    break
        ret[node, :n_cell_edges] = sorted_edges[:n_cell_edges, 0]

    return ret