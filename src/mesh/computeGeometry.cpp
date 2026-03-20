#include "mesh.h"
#include "computeGeometry.h"
#include <algorithm>

#include "mpi.h"
#include "halo.h"
#include "unordered_set"

void haloExchangeCellGeom(Mesh& local, MPI_Comm comm)
{
    // 2 components: Cx, Cy
    int ncomp = 2;
    std::vector<double> buf((size_t)local.nCells * ncomp, 0.0);

    for (int lc = 0; lc < local.nCells; ++lc) {
        buf[(size_t)lc * ncomp + 0] = local.cells[lc].C.x;
        buf[(size_t)lc * ncomp + 1] = local.cells[lc].C.y;
    }

    haloExchangeCellData(local, buf.data(), ncomp, comm);

    // only ghost region needs to be updated back
    for (int lc = local.nOwned; lc < local.nCells; ++lc) {
        local.cells[lc].C.x = buf[(size_t)lc * ncomp + 0];
        local.cells[lc].C.y = buf[(size_t)lc * ncomp + 1];
        // printf("%.3e \n ",local.cells[lc].C.x);
    }
}

void addBoundaryPseudoCells(Mesh& mesh) {
    for (int bf = 0; bf < mesh.nBoundaryFaces; ++bf) {
        int faceI = mesh.nInternalFaces + bf;
        mesh.faces[faceI].neigh = mesh.nCells + bf; // boundary pseudo cell
    }
}

void computeGeometry(Mesh& mesh,MPI_Comm comm){
    // 计算几何量（单元中心、面中心、面积矢量、面积大小、单元中心到面中心的矢量）
    calC(mesh);
    calV(mesh);
    calSf(mesh);
    calCf(mesh);

    haloExchangeCellGeom(mesh, comm);
    // TODO: after get the center position of ghost cells
    calEf(mesh);
    calGC(mesh);

    // Tf计算
    for(size_t faceI=0;faceI<mesh.nFaces;faceI++){
        mesh.faces[faceI].Tf= mesh.faces[faceI].Sf - mesh.faces[faceI].Ef;
    }

    // 最小二乘法计算梯度时需要的逆矩阵
    for(size_t cellI=0;cellI<mesh.nCells;cellI++){
        double A00=0, A01=0, A11=0;
        auto& cell = mesh.cells[cellI];
        for(auto faceI:cell.cell2face){
            vector dC;
            size_t owner=mesh.faces[faceI].owner;
            size_t neigh=mesh.faces[faceI].neigh;
            if(cellI==owner){
                if(neigh<mesh.nCells){ // 内部面
                    dC= mesh.cells[neigh].C-mesh.cells[owner].C;
                }else{ // 边界面，邻居是伪单元，使用边界面中心代替邻居单元中心
                    dC= mesh.faces[faceI].Cf - mesh.cells[owner].C;
                }
            }else{
                dC= mesh.cells[owner].C-mesh.cells[neigh].C;
            }
            A00 += dC.x * dC.x;
            A01 += dC.y * dC.x;
            A11 += dC.y * dC.y;
        }
        double det =A00*A11-A01*A01;
        if(std::abs(det)>1e-20){
            cell.invA[0] = A11/det;
            cell.invA[1] = -A01/det;
            cell.invA[2] = A00/det;
        }else{
            cell.invA[0] = 0.0;
            cell.invA[1] = 0.0;
            cell.invA[2] = 0.0;
            printf("Warning: Singular matrix encountered when computing gradient at cell %zu\n", cellI);
        }
    }
    // 计算单元到单元的距离
    for(size_t cellI=0;cellI<mesh.nCells;cellI++){
        auto& cell = mesh.cells[cellI];
        for(auto faceI:cell.cell2face){
            size_t owner=mesh.faces[faceI].owner;
            size_t neigh=mesh.faces[faceI].neigh;
            vector dC;
            if(cellI==owner){
                if(neigh<mesh.nCells){
                    dC= mesh.cells[neigh].C-mesh.cells[owner].C;
                }else{
                    dC= mesh.faces[faceI].Cf - mesh.cells[owner].C;
                }
            }else{
                dC= mesh.cells[owner].C-mesh.cells[neigh].C;
            }
            cell.dxyz.push_back(std::array<double,2>{dC.x,dC.y});
        }
    }
}
void buildCellFaces(Mesh& mesh)
{
    // 根据owner/neighbour数据构建单元的面列表
    for(int facei=0;facei<mesh.nFaces;facei++){
        int owner = mesh.faces[facei].owner;
        int neigh = mesh.faces[facei].neigh;
        if(owner!=-1){
            mesh.cells[owner].cell2face.push_back(facei);
        }
        if(neigh!=-1){
            mesh.cells[neigh].cell2face.push_back(facei);
        }
    }
}
void buildCellPoint(Mesh& mesh) {
    mesh.cell2node.resize(mesh.nOwned);
    for (size_t cellI = 0; cellI < mesh.nOwned; ++cellI) {
        std::unordered_set<size_t> uniqueNodes; // 避免重复点
        for(int faceIdx : mesh.cells[cellI].cell2face){
            uniqueNodes.insert(mesh.faces[faceIdx].n1);
            uniqueNodes.insert(mesh.faces[faceIdx].n2);
        }
        mesh.cell2node[cellI] = std::vector<int>(uniqueNodes.begin(), uniqueNodes.end());
    }
}
void adjustCell2node(Mesh& mesh) {
    // 将cell2node_转换为逆时针排序（为了使用公式计算体积）
    auto& cell2node = mesh.cell2node;
    const auto& cells = mesh.cells;
    const auto& points = mesh.points;
    for (size_t cellI = 0; cellI < mesh.nOwned; ++cellI) {
        std::vector<int> nodes(cell2node[cellI].begin(), cell2node[cellI].end());
        std::sort(nodes.begin(), nodes.end(),
            [&](size_t a, size_t b) {
                const vector& C =  cells[cellI].C; // 单元质心
                const vector& Pa = points[a];
                const vector& Pb = points[b];

                // 计算相对质心的角度
                double angleA = std::atan2(Pa.y - C.y, Pa.x - C.x);
                double angleB = std::atan2(Pb.y - C.y, Pb.x - C.x);

                return angleA < angleB; // 逆时针排序
            }
        );
        cell2node[cellI] = std::move(nodes);
    }
}

void calSf(Mesh& mesh) {
    auto& faces =mesh.faces;
    const auto& points = mesh.points;

    for (size_t faceI = 0; faceI < faces.size(); ++faceI) {
        const vector& p0 = points[faces[faceI].n1];
        const vector& p1 = points[faces[faceI].n2];

        // 计算边矢量
        vector edge = p1 - p0;

        // 计算面法向量（二维中旋转90度）
        mesh.faces[faceI].Sf = vector(edge.y, -edge.x, 0);
    }
}
void calCf(Mesh& mesh) {
    auto& faces = mesh.faces;
    auto& points = mesh.points;

    for (size_t faceI = 0; faceI < faces.size(); ++faceI) {
        const auto& face = faces[faceI];
        faces[faceI].Cf.x = (points[face.n1].x + points[face.n2].x) / 2.0;
        faces[faceI].Cf.y = (points[face.n1].y + points[face.n2].y) / 2.0;
    }
}
void calV(Mesh& mesh) {
    // 保证cell2node是逆时针排序的（为了使用公式计算体积）·
    adjustCell2node(mesh);
    const auto& cell2node = mesh.cell2node;
    const auto& points = mesh.points;
    for (size_t cellI = 0; cellI < mesh.nOwned; ++cellI) {

        double area = 0.0;
        for(int i=0;i<cell2node[cellI].size();i++){
            const auto& p1 = points[cell2node[cellI][i]];
            const auto& p2 = points[cell2node[cellI][(i+1)%cell2node[cellI].size()]];
            area += (p1.x*p2.y - p2.x*p1.y); // Shoelace公式
        }
        mesh.cells[cellI].V = std::abs(area) / 2.0; // 取绝对值并除以2
    }
}
void calC(Mesh& mesh) {
    for (size_t cellI = 0; cellI < mesh.nOwned; ++cellI) {
        double Cx = 0.0, Cy = 0.0;
        for (size_t nodeIdx : mesh.cell2node[cellI]) {
            Cx += mesh.points[nodeIdx].x;
            Cy += mesh.points[nodeIdx].y;
        }
        mesh.cells[cellI].C.x = Cx / static_cast<double>(mesh.cell2node[cellI].size());
        mesh.cells[cellI].C.y = Cy / static_cast<double>(mesh.cell2node[cellI].size());
    }
}
void calEf(Mesh& mesh){
    // 内部面
    for(int facei=0;facei<mesh.nInternalFaces;facei++){
        int owner = mesh.faces[facei].owner;
        int neigh = mesh.faces[facei].neigh;
        const vector& Sf = mesh.faces[facei].Sf;
        double magSf= Sf.mag();
        vector ef = mesh.cells[neigh].C - mesh.cells[owner].C;
        double magEf= ef.mag();
        ef.x /= magEf;
        ef.y /= magEf;
        double dot= ef.x*Sf.x+ef.y*Sf.y;
        if(dot<=0.0){
            std::cout<<"faceI: "<<facei<<" dot< 0 "<<std::endl;
        }
        mesh.faces[facei].Ef.x= ef.x*magSf*magSf/dot;
        mesh.faces[facei].Ef.y= ef.y*magSf*magSf/dot;
    }
    // boundary
    for(int facei=mesh.nInternalFaces;facei<mesh.nFaces;facei++){
        int owner = mesh.faces[facei].owner;
        const vector& Sf = mesh.faces[facei].Sf;
        double magSf= Sf.mag();
        vector ef = mesh.faces[facei].Cf - mesh.cells[owner].C;
        double magEf= ef.mag();
        ef.x /= magEf;
        ef.y /= magEf;
        double dot= ef.x*Sf.x+ef.y*Sf.y;
        if(dot<=0.0){
            std::cout<<"faceI: "<<facei<<" dot< 0 "<<std::endl;
        }
        mesh.faces[facei].Ef.x= ef.x*magSf*magSf/dot;
        mesh.faces[facei].Ef.y= ef.y*magSf*magSf/dot;
    }
}
void calGC(Mesh& mesh) {
    // 内部面
    for(int facei=0;facei<mesh.nInternalFaces;facei++){
        int owner = mesh.faces[facei].owner;
        int neigh = mesh.faces[facei].neigh;
        const vector& Sf = mesh.faces[facei].Sf;
        double magSf= Sf.mag();
        vector Cf = mesh.faces[facei].Cf;
        vector C_owner= mesh.cells[owner].C;
        vector C_neigh= mesh.cells[neigh].C;

        vector rCF= C_neigh - C_owner;
        double magrCF= rCF.mag();

        vector rCf= Cf - C_owner;
        double dot= rCf.x*rCF.x + rCf.y*rCF.y;

        double beta=dot/magrCF/magrCF;
        if(beta<=0.0 || beta>=1.0){
            std::cout<<"faceI: "<<facei<<"beta: "<<beta<<std::endl;
        }
        mesh.faces[facei].gc= 1.0 - beta;
    }
}