#ifndef MESH_H_INCLUDED
#define MESH_H_INCLUDED

#include <iostream>
#include <vector>
#include <array>
#include <string.h>
#include "defs.h"
#include <cmath>
#include <unordered_map>

class vector
{
public:
    double x, y, z;
    
    vector() : x(0), y(0), z(0) {}
    vector(double x, double y, double z) : x(x), y(y), z(z) {}
    
    vector operator-(const vector& v) const {
        return vector(x - v.x, y - v.y, z - v.z);
    }
    
    vector operator+(const vector& v) const {
        return vector(x + v.x, y + v.y, z + v.z);
    }
    
    vector operator*(double s) const {
        return vector(x * s, y * s, z * s);
    }
    vector operator/(double s) const {
        if (s == 0.0) {
            throw std::invalid_argument("division by 0");
            return vector(0,0,0);
        }
        else {
            return vector(x / s, y / s, z / s);
        }
    }
    
    double mag() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    
    vector unit() const {
        double m = mag();
        return m > 0 ? vector(x/m, y/m, z/m) : vector();
    }
};

struct Face
{
    int n1=-1,n2=-1;

    int owner=-1;
    int neigh=-1;

    BCType bc_type;

    // geometry 
    vector Cf;
    vector Sf,Ef,Tf;
    double gc=0;
};

struct Cell
{
    std::vector<int> cell2face;
    // geometry
    vector C;
    double V=0;
    double invA[3]={0.0};
    std::vector<std::array<double,2>> dxyz; // 单元到单元的距离矢量
};

struct HaloPlan
{
    std::vector<int> neighbors;// neighbor ranks

    std::vector<int> send_offsets;
    std::vector<int> recv_offsets;// 

    std::vector<int> send_cells; // local owned cell ids
    std::vector<int> recv_cells; // local ghost cell ids

    std::vector<int> ghost_owner_rank; // size = nCells - nOwned
};

struct CellFaceInfo {
    int nf;
    int neigh_id[8];
    int nb_base[8];
    double sfx[8], sfy[8];
    int8_t orient[8];
};

struct Mesh
{
    std::vector<vector> points;
    std::vector<Face> faces;
    std::vector<Cell> cells;
    std::vector<std::vector<int>> cell2node;

    std::vector<int> interiorCells;
    std::vector<int> boundaryCells;

    std::vector<int> part;

    std::vector<int> l2g_point, l2g_cell;
    std::unordered_map<int,int> g2l_point, g2l_cell;

    HaloPlan halo;

    // boundary face 在最后面
    int nInternalFaces = 0;
    int nBoundaryFaces = 0;
    int nFaces = 0;
    int nCells = 0;
    int nOwned = 0;
};

bool parseFluentFile(const std::string& filePath,Mesh& mesh);
void parseNodesSection(Mesh& mesh,std::istream& file,std::string& line);
void parseFacesSection(Mesh& mesh,std::istream& file,std::string& line);
void parseCellsSection(Mesh& mesh,std::istream& file,std::string& line);

// void printfInfo(Mesh& mesh);

#endif