#include "partition.h"

#include "mesh.h"
#include <metis.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "computeGeometry.h"
#include "halo.h"

static bool build_local_pack_for_rank(
    LocalPack& pack,
    const Mesh& mesh,
    int target_rank)
{
    pack.faces.clear();
    pack.pts.clear();
    pack.cell_gids.clear();
    pack.ghost_owner_ranks.clear();
    pack.owned_count = 0;

    const auto& part = mesh.part;
    if ((int)part.size() != mesh.nCells) {
        return false;
    }

    // 1) owned cells
    for (int c = 0; c < mesh.nCells; ++c) {
        if (part[c] == target_rank) {
            pack.cell_gids.push_back(c);
        }
    }
    pack.owned_count = (int)pack.cell_gids.size();

    // 2) scan faces, collect faces and ghosts
    std::unordered_set<int> ghost_set;
    ghost_set.reserve(pack.owned_count > 0 ? pack.owned_count : 8);

    for (int facei = 0; facei < (int)mesh.faces.size(); ++facei) {
        const auto& face = mesh.faces[facei];

        int ro = (face.owner >= 0) ? part[face.owner] : -1;
        int rn = (face.neigh >= 0) ? part[face.neigh] : -1;

        bool add_face = false;

        // owner side belongs to target_rank
        if (ro == target_rank) {
            add_face = true;
        }
        // cross-partition internal face, neighbor side belongs to target_rank
        else if (face.neigh >= 0 && rn == target_rank && rn != ro) {
            add_face = true;
        }

        if (!add_face) continue;

        FacePack fp;
        fp.gFace = facei;
        fp.n1 = face.n1;
        fp.n2 = face.n2;
        fp.owner = face.owner;
        fp.neigh = face.neigh;
        fp.bc = (int)face.bc_type;
        pack.faces.push_back(fp);

        // cross-partition internal face: collect ghost
        if (face.neigh >= 0 && ro != rn) {
            if (ro == target_rank) {
                ghost_set.insert(face.neigh);
            } else if (rn == target_rank) {
                ghost_set.insert(face.owner);
            }
        }
    }

    // 3) append ghosts after owned
    std::vector<int> ghosts(ghost_set.begin(), ghost_set.end());
    std::sort(ghosts.begin(), ghosts.end());

    pack.cell_gids.reserve(pack.owned_count + (int)ghosts.size());
    pack.ghost_owner_ranks.reserve(ghosts.size());

    for (int g : ghosts) {
        pack.cell_gids.push_back(g);
        pack.ghost_owner_ranks.push_back(part[g]);
    }

    // 4) build pts from faces using sparse set
    std::unordered_set<int> used_points;
    used_points.reserve(pack.faces.size() * 2);

    for (const auto& f : pack.faces) {
        used_points.insert(f.n1);
        used_points.insert(f.n2);
    }

    pack.pts.clear();
    pack.pts.reserve(used_points.size());
    for (int gp : used_points) {
        const auto& p = mesh.points[gp];
        PointPack pp;
        pp.gPoint = gp;
        pp.x = p.x;
        pp.y = p.y;
        pp.z = p.z;
        pack.pts.push_back(pp);
    }

    return true;
}
static void buildLocalPoints(const LocalPack& pack,
    Mesh& local)
{
    auto& g2l_point = local.g2l_point;
    auto& l2g_point  = local.l2g_point;
    const auto& pts = pack.pts;
    local.points.clear();
    local.points.reserve(pts.size());
    l2g_point.clear();
    l2g_point.reserve(pts.size());
    g2l_point.clear();
    g2l_point.reserve(pts.size()*2);

    for(int i=0;i<(int)pts.size();++i){
        local.points.push_back(vector(pts[i].x, pts[i].y, pts[i].z));
        l2g_point.push_back(pts[i].gPoint);
        g2l_point[pts[i].gPoint] = i;
    }
}
static void buildLocalCells(const LocalPack& pack,
    Mesh& local)
{
    auto& g2l_cell = local.g2l_cell;
    auto& l2g_cell = local.l2g_cell;
    const auto& cell_gids = pack.cell_gids;
    local.cells.clear();
    local.cells.resize(cell_gids.size()); // only owned
    local.nOwned = pack.owned_count;
    local.nCells = (int)cell_gids.size();

    l2g_cell = cell_gids;
    g2l_cell.clear();
    g2l_cell.reserve(local.nCells*2);
    for(int lc=0; lc<(int)local.nCells; ++lc){
        g2l_cell[ cell_gids[lc] ] = lc;
    }
}
static void buildLocalFaces(const LocalPack& pack,
    Mesh& local)
{
    auto& g2l_cell = local.g2l_cell;
    auto& g2l_point = local.g2l_point; 

    const auto& facePacks = pack.faces;
    local.faces.clear();
    local.faces.reserve(facePacks.size());

    int internalCount = 0;
    int boundaryCount = 0;

    for(const auto& fp : facePacks){
        Face f;
        f.n1 = g2l_point.at(fp.n1);
        f.n2 = g2l_point.at(fp.n2);
        f.bc_type = (BCType)fp.bc;
        
        auto itO = g2l_cell.find(fp.owner);
        if(itO == g2l_cell.end()){
            printf("Error: can't find local owner in face\n");
            continue;
        }
        f.owner = itO->second;

        if(fp.neigh>=0){
            auto itN = g2l_cell.find(fp.neigh);
            if(itN == g2l_cell.end()){
                printf("Error: can't find local neigh in face\n");
                continue;
            }
            f.neigh = itN->second;
            internalCount++;
        }else{
            f.neigh = -1;
            boundaryCount++;
        }
        local.faces.push_back(f);
    }

    local.nFaces = (int)local.faces.size();
    local.nInternalFaces = internalCount;
    local.nBoundaryFaces = boundaryCount;
}
static void send_pack(const LocalPack& pack, int dest, MPI_Comm comm) {
    int counts[5];
    counts[0] = (int)pack.cell_gids.size();
    counts[1] = (int)pack.faces.size();
    counts[2] = (int)pack.pts.size();
    counts[3] = (int)pack.owned_count;
    counts[4] = (int)pack.ghost_owner_ranks.size();

    MPI_Send(counts, 5, MPI_INT, dest, 100, comm);

    if (counts[0] > 0)
        MPI_Send((void*)pack.cell_gids.data(), counts[0], MPI_INT, dest, 101, comm);

    if (counts[1] > 0)
        MPI_Send((void*)pack.faces.data(), counts[1] * (int)sizeof(FacePack),
                 MPI_BYTE, dest, 102, comm);

    if (counts[2] > 0)
        MPI_Send((void*)pack.pts.data(), counts[2] * (int)sizeof(PointPack),
                 MPI_BYTE, dest, 103, comm);

    if (counts[4] > 0)
        MPI_Send((void*)pack.ghost_owner_ranks.data(), counts[4], MPI_INT, dest, 104, comm);
}

static void recv_pack(LocalPack& pack, MPI_Comm comm) {
    int counts[5];
    MPI_Recv(counts, 5, MPI_INT, 0, 100, comm, MPI_STATUS_IGNORE);

    pack.cell_gids.resize(counts[0]);
    pack.faces.resize(counts[1]);
    pack.pts.resize(counts[2]);
    pack.owned_count = counts[3];
    pack.ghost_owner_ranks.resize(counts[4]);

    if (counts[0] > 0)
        MPI_Recv(pack.cell_gids.data(), counts[0], MPI_INT, 0, 101, comm, MPI_STATUS_IGNORE);

    if (counts[1] > 0)
        MPI_Recv(pack.faces.data(), counts[1] * (int)sizeof(FacePack),
                 MPI_BYTE, 0, 102, comm, MPI_STATUS_IGNORE);

    if (counts[2] > 0)
        MPI_Recv(pack.pts.data(), counts[2] * (int)sizeof(PointPack),
                 MPI_BYTE, 0, 103, comm, MPI_STATUS_IGNORE);

    if (counts[4] > 0)
        MPI_Recv(pack.ghost_owner_ranks.data(), counts[4], MPI_INT, 0, 104, comm, MPI_STATUS_IGNORE);
}

static void build_adjacency(const Mesh& mesh,
    std::vector<idx_t>& xadj,
    std::vector<idx_t>& adjncy) {
    std::vector<std::vector<int>> adj(mesh.cells.size());
    for (const auto& face : mesh.faces) {
        if (face.neigh >= 0) {
            adj[face.owner].push_back(face.neigh);
            adj[face.neigh].push_back(face.owner);
        }
    }
    xadj.resize(mesh.cells.size() + 1);
    size_t total = 0;
    for (size_t i = 0; i < adj.size(); ++i) {
        auto& a = adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        xadj[i] = static_cast<idx_t>(total);
        total += a.size();
    }
    xadj[adj.size()] = static_cast<idx_t>(total);
    adjncy.resize(total);
    size_t pos = 0;
    for (size_t i = 0; i < adj.size(); ++i) {
        for (int v : adj[i]) {
            adjncy[pos++] = static_cast<idx_t>(v);
        }
    }
}

bool partition_and_distribute(
	Mesh& mesh, 
	Mesh& local,
	MPI_Comm comm, std::string& err)
{   
    int rank = 0;
	int size = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    std::vector<int> ghost_owner_ranks;
    if(rank==0){
        // partition(simple partition)
        auto& part = mesh.part;
        part.resize(mesh.nCells);
        if(size==1){
            std::fill(part.begin(),part.end(),0);
        }else{
            std::vector<idx_t> xadj;
            std::vector<idx_t> adjncy;
            build_adjacency(mesh, xadj, adjncy);
            idx_t n = static_cast<idx_t>(mesh.cells.size());
            idx_t ncon = 1;
            idx_t nparts = static_cast<idx_t>(size);
            idx_t objval = 0;
            std::vector<idx_t> part_metis(n);

            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;   // 更偏向通信体积
            options[METIS_OPTION_CONTIG]  = 1;                  // 尽量连通
            options[METIS_OPTION_MINCONN] = 1;                  // 减少子域连接数
            options[METIS_OPTION_NUMBERING] = 0;

            int status = METIS_PartGraphKway(&n, &ncon, xadj.data(), adjncy.data(),
            nullptr, nullptr, nullptr, &nparts,
            nullptr, nullptr, options, &objval, part_metis.data());
            if (status != METIS_OK) {
                err = "METIS_PartGraphKway failed.";
                return false;
            }
            for(int cellI=0;cellI<mesh.nCells;cellI++){
                part[cellI]=part_metis[cellI];
            }
        }
        // for(int c=0;c<mesh.nCells;c++){
        //     part[c] = ownerRank_block(c, mesh.nCells, size);
        // }

        // send to each rank
        for(int r=0;r<size;r++){
            LocalPack pack;
            if (!build_local_pack_for_rank(pack, mesh, r)) {
                err = "build_local_pack_for_rank failed for rank " + std::to_string(r);
                return false;
            }
            if(r==0){
                ghost_owner_ranks = pack.ghost_owner_ranks;
                buildLocalPoints(pack, local);
                buildLocalCells(pack, local);
                buildLocalFaces(pack, local);
                
            }else{
                send_pack(pack,r,comm);
            }
        }
    }else{
        LocalPack pack;
        recv_pack(pack,comm);
        ghost_owner_ranks = pack.ghost_owner_ranks;
        buildLocalPoints(pack, local);
        buildLocalCells(pack, local);
        buildLocalFaces(pack, local);
    }
    // build halo
    buildHaloPlan(local, ghost_owner_ranks, comm);
    // compute geometry
    buildCellFaces(local);// cell2face
    buildCellPoint(local);// cell2node
    addBoundaryPseudoCells(local);// adjust neigh for boundary faces
    computeGeometry(local,comm);
    buildInteriorBoundaryCells(local);

    return true;
}