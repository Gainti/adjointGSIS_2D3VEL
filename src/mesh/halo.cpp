#include "halo.h"

#include <algorithm>
#include <memory.h>

bool buildHaloPlan(Mesh& local, const std::vector<int>& ghost_owner_ranks, MPI_Comm comm)
{
    HaloPlan& hp = local.halo;
    hp = HaloPlan();// clean

    const int nOwned = local.nOwned;
    const int nCells = local.nCells;
    const int nGhost = nCells - nOwned;

    // -----------------------------
    // 1) build recv side
    // -----------------------------
    std::unordered_map<int, std::vector<int>> recv_map; // owner_rank -> local ghost cell ids
    std::unordered_map<int, std::vector<int>> recv_gids_map; // owner_rank -> global cell ids of ghosts

    for (int ig = 0; ig < nGhost; ++ig) {
        int lc = nOwned + ig;
        int owner_rank = ghost_owner_ranks[ig];
        int gCell = local.l2g_cell[lc];

        recv_map[owner_rank].push_back(lc);
        recv_gids_map[owner_rank].push_back(gCell);
    }

    // neighbors in sorted order for determinism
    std::vector<int> neighbors;
    neighbors.reserve(recv_map.size());
    for (auto& kv : recv_map) neighbors.push_back(kv.first);
    std::sort(neighbors.begin(), neighbors.end());

    hp.neighbors = neighbors;

    hp.recv_offsets.resize(neighbors.size() + 1, 0);
    for (int k = 0; k < (int)neighbors.size(); ++k) {
        int r = neighbors[k];
        hp.recv_offsets[k + 1] = hp.recv_offsets[k] + (int)recv_map[r].size();
        for (int lc : recv_map[r]) {
            hp.recv_cells.push_back(lc);
        }
    }

    // -----------------------------
    // 2) handshake: ask owner ranks which global cells we need
    // -----------------------------
    std::vector<int> send_counts(neighbors.size(), 0), recv_counts(neighbors.size(), 0);
    std::vector<MPI_Request> reqs;

    reqs.reserve(2 * neighbors.size());

    for (int k = 0; k < (int)neighbors.size(); ++k) {
        send_counts[k] = (int)recv_gids_map[neighbors[k]].size();
        MPI_Request req;
        MPI_Isend(&send_counts[k], 1, MPI_INT, neighbors[k], 200, comm, &req);
        reqs.push_back(req);
    }
    for (int k = 0; k < (int)neighbors.size(); ++k) {
        MPI_Request req;
        MPI_Irecv(&recv_counts[k], 1, MPI_INT, neighbors[k], 200, comm, &req);
        reqs.push_back(req);
    }
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // send the global cell ids we need
    std::vector<std::vector<int>> incoming_gids(neighbors.size());
    for (int k = 0; k < (int)neighbors.size(); ++k) {
        int nb = neighbors[k];
        incoming_gids[k].resize(recv_counts[k]);

        MPI_Request req1, req2;
        auto& out = recv_gids_map[nb];

        if (!out.empty())
            MPI_Isend(out.data(), (int)out.size(), MPI_INT, nb, 201, comm, &req1);
        else
            MPI_Isend(nullptr, 0, MPI_INT, nb, 201, comm, &req1);

        if (!incoming_gids[k].empty())
            MPI_Irecv(incoming_gids[k].data(), (int)incoming_gids[k].size(), MPI_INT, nb, 201, comm, &req2);
        else
            MPI_Irecv(nullptr, 0, MPI_INT, nb, 201, comm, &req2);

        reqs.push_back(req1);
        reqs.push_back(req2);
    }
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    reqs.clear();

    // -----------------------------
    // 3) build send side: map requested global ids to my local owned ids
    // -----------------------------
    hp.send_offsets.resize(neighbors.size() + 1, 0);

    for (int k = 0; k < (int)neighbors.size(); ++k) {
        for (int gCell : incoming_gids[k]) {
            auto it = local.g2l_cell.find(gCell);
            if (it == local.g2l_cell.end()) {
                printf("Error: requested global cell %d not found in local.g2l_cell\n",gCell);
                MPI_Abort(comm, 1);
                return false;
            }
            int lc = it->second;
            if (lc >= local.nOwned) {
                printf("Error: requested global cell %d is is not owned locally\n",gCell);
                MPI_Abort(comm, 1);
                return false;
            }
            hp.send_cells.push_back(lc);
        }
        hp.send_offsets[k + 1] = (int)hp.send_cells.size();
    }
    
    return true;
}


void haloExchangeCellData(
    const Mesh& local,
    double* data,
    int ncomp,
    MPI_Comm comm)
{
    const HaloPlan& hp = local.halo;
    const int nnei = (int)hp.neighbors.size();

    std::vector<std::vector<double>> send_bufs(nnei), recv_bufs(nnei);
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * nnei);

    // post receives
    for (int k = 0; k < nnei; ++k) {
        int nb = hp.neighbors[k];
        int nrecv = hp.recv_offsets[k + 1] - hp.recv_offsets[k];
        recv_bufs[k].resize((size_t)nrecv * ncomp);

        MPI_Request req;
        MPI_Irecv(recv_bufs[k].data(), (int)recv_bufs[k].size(), MPI_DOUBLE,
                  nb, 300, comm, &req);
        reqs.push_back(req);
    }

    // pack and send
    for (int k = 0; k < nnei; ++k) {
        int nb = hp.neighbors[k];
        int nsend = hp.send_offsets[k + 1] - hp.send_offsets[k];
        send_bufs[k].resize((size_t)nsend * ncomp);

        for (int i = 0; i < nsend; ++i) {
            int lc = hp.send_cells[hp.send_offsets[k] + i];
            memcpy(&send_bufs[k][(size_t)i * ncomp],
                        &data[(size_t)lc * ncomp],
                        sizeof(double) * ncomp);
        }

        MPI_Request req;
        MPI_Isend(send_bufs[k].data(), (int)send_bufs[k].size(), MPI_DOUBLE,
                  nb, 300, comm, &req);
        reqs.push_back(req);
    }

    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // unpack into ghosts
    for (int k = 0; k < nnei; ++k) {
        int nrecv = hp.recv_offsets[k + 1] - hp.recv_offsets[k];
        for (int i = 0; i < nrecv; ++i) {
            int lc = hp.recv_cells[hp.recv_offsets[k] + i];
            memcpy(&data[(size_t)lc * ncomp],
                        &recv_bufs[k][(size_t)i * ncomp],
                        sizeof(double) * ncomp);
        }
    }
}

// template <typename T>
// void haloExchangeCellDataT(
//     const Mesh& local,
//     T* data,
//     int ncomp,
//     MPI_Datatype mpi_type,
//     MPI_Comm comm)
// {
//     const HaloPlan& hp = local.halo;
//     const int nnei = (int)hp.neighbors.size();

//     std::vector<std::vector<T>> send_bufs(nnei), recv_bufs(nnei);
//     std::vector<MPI_Request> reqs;
//     reqs.reserve(2 * nnei);

//     for (int k = 0; k < nnei; ++k) {
//         int nb = hp.neighbors[k];
//         int nrecv = hp.recv_offsets[k + 1] - hp.recv_offsets[k];
//         recv_bufs[k].resize((size_t)nrecv * ncomp);

//         MPI_Request req;
//         MPI_Irecv(recv_bufs[k].data(), (int)recv_bufs[k].size(), mpi_type,
//                   nb, 300, comm, &req);
//         reqs.push_back(req);
//     }

//     for (int k = 0; k < nnei; ++k) {
//         int nb = hp.neighbors[k];
//         int nsend = hp.send_offsets[k + 1] - hp.send_offsets[k];
//         send_bufs[k].resize((size_t)nsend * ncomp);

//         for (int i = 0; i < nsend; ++i) {
//             int lc = hp.send_cells[hp.send_offsets[k] + i];
//             memcpy(&send_bufs[k][(size_t)i * ncomp],
//                         &data[(size_t)lc * ncomp],
//                         sizeof(T) * ncomp);
//         }

//         MPI_Request req;
//         MPI_Isend(send_bufs[k].data(), (int)send_bufs[k].size(), mpi_type,
//                   nb, 300, comm, &req);
//         reqs.push_back(req);
//     }

//     if (!reqs.empty()) {
//         MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
//     }

//     for (int k = 0; k < nnei; ++k) {
//         int nrecv = hp.recv_offsets[k + 1] - hp.recv_offsets[k];
//         for (int i = 0; i < nrecv; ++i) {
//             int lc = hp.recv_cells[hp.recv_offsets[k] + i];
//             memcpy(&data[(size_t)lc * ncomp],
//                         &recv_bufs[k][(size_t)i * ncomp],
//                         sizeof(T) * ncomp);
//         }
//     }
// }