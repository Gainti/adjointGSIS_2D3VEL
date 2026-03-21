#pragma once

// TODO: 针对边界节点进行参数化，一方面能够给定参数计算出边界节点的位移，另一方面要给出边界节点的位移关于参数的导数


// struct DesignVariables {
//     std::vector<double> x;
// };

// 根据参数来移动网格节点坐标
// void applyParameterization(Mesh& mesh, const DesignVariables& dv);

// 给出边界节点的位移关于参数的导数
// 接口应该是目标函数关于节点的导数，输出应该是目标函数关于参数的导数
// void computeParameterizationDerivative(
//     const Mesh& mesh,
//     const DesignVariables& dv,
//     std::vector<std::vector<vector>>& dXdv);