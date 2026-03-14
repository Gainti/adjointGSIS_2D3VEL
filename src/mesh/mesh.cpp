#include "mesh.h"
#include "defs.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <memory>
#include <sstream>
#include <cctype>

// 十六进制字符串转十进制整数
size_t str2num(std::string str)
{
    return stoi(str, nullptr, 16);
}
// 解析Fluent网格文件
bool parseFluentFile(const std::string& filePath,Mesh& mesh) {
    std::fstream file;
    file.open(filePath, std::ios::in);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // 识别网格部分
        if (line.find("(10") == 0) {         // 节点部分
            parseNodesSection(mesh,file,line);
        } else if (line.find("(13") == 0) {   // 面部分
            parseFacesSection(mesh,file,line);
        } else if (line.find("(12") == 0) {   // 单元部分
            parseCellsSection(mesh,file,line);
        }
    }
    file.close();
    // 计算内部面数量
    mesh.nInternalFaces = 0;
    mesh.nBoundaryFaces = 0;
    mesh.nFaces = mesh.faces.size();
    for(auto facei : mesh.faces){
        if(facei.neigh>=0){
            mesh.nInternalFaces++;
        }else{
            mesh.nBoundaryFaces++;
        }
    }
    mesh.nCells = mesh.cells.size();
    mesh.nOwned = mesh.nCells;
    if(mesh.nInternalFaces+mesh.nBoundaryFaces!=mesh.nFaces){
        printf("Error: Internal faces + Boundary faces does not equal total faces!\n\n");
        return false;
    }
    return true;
}
void parseCellsSection(Mesh& mesh,std::istream& file,std::string& line) {
    // 解析单元头信息 (12 (zone-id first-index last-index type element-type))
    size_t zoneId,temp;
    char dummy;
    std::string firstStr, lastStr;

    std::istringstream iss(line);
    iss >> dummy >> temp>>dummy>>zoneId >> firstStr >> lastStr;
    if (zoneId==0) {
        size_t firstIdx = str2num(firstStr);
        size_t lastIdx = str2num(lastStr);

        mesh.cells.resize(lastIdx-firstIdx+1);
    }
}
void parseNodesSection(Mesh& mesh,std::istream& file,std::string& line)
{
    // 解析节点头信息 (10 (zone-id first-index last-index type ND))
    size_t zoneId, type, nd,temp;
    char dummy;
    std::string firstStr, lastStr;

    std::istringstream iss(line);
    iss >> dummy >> temp>>dummy>>zoneId >> firstStr >> lastStr;

    if (zoneId!=0) {
        iss>>type >> nd >> dummy;

        size_t firstIdx = str2num(firstStr);
        size_t lastIdx = str2num(lastStr);

        // 读取节点坐标
        for (size_t i = firstIdx; i <= lastIdx; ++i) {
            std::getline(file, line);
            double x, y, z = 0;
            std::istringstream coordStream(line);
            coordStream >> x >> y;
            if (nd == 3) coordStream >> z;
            // TODO: 3d
            mesh.points.push_back(vector(x, y, z));
        }
    }
}
void parseFacesSection(Mesh& mesh,std::istream& file,std::string& line)
{
    // 解析面头信息 (13 (zone_id first_index last_index bc_type face_type))
    size_t zoneId, faceType,temp;
    char dummy;
    std::string firstStr, lastStr;
    std::string bcStr;

    std::istringstream iss(line);
    iss >> dummy >> temp >>dummy>>zoneId >> firstStr >> lastStr ;

    //
    if (zoneId!=0) {
        iss>>bcStr >> faceType >> dummy;
        size_t firstIdx = str2num(firstStr);
        size_t lastIdx = str2num(lastStr);
        size_t bcType = str2num(bcStr);

        // 读取面数据
        for (size_t i = firstIdx; i <= lastIdx; ++i) {
            Face face;
            std::getline(file, line);
            std::istringstream faceStream(line);

            std::vector<size_t> faceNodes;

            // 根据面类型处理不同格式
            if (faceType == 0 || faceType == 5) { // 混合或多边形面
                size_t numNodes;
                faceStream >> numNodes;
                faceNodes.resize(numNodes);
                for (size_t j = 0; j < numNodes; ++j) {
                    std::string nodeStr;
                    faceStream >> nodeStr;
                    faceNodes[j] = str2num(nodeStr);
                    faceNodes[j]--; // 转换为0-based索引
                }
            } else { // 标准面类型
                std::string nodeStr;
                while (faceStream >> nodeStr) {
                    size_t node= str2num(nodeStr);
                    faceNodes.push_back(node - 1); // 转换为0-based索引
                }
            }
            // 所有者/邻居单元 (最后两个值)
            size_t neighbour = faceNodes.back(); faceNodes.pop_back();
            size_t owner = faceNodes.back(); faceNodes.pop_back();

            face.owner=owner;
            face.neigh=neighbour == 0 ? -1 : neighbour;
            face.n1=faceNodes[0];
            face.n2=faceNodes[1];
            face.bc_type=(BCType) bcType;
            mesh.faces.push_back(face);
        }
    }
}