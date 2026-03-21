#include "optimize_driver.h"

// void runOptimize(const SolverConfig& cfg)
// {
//     DesignVariables dv;
//     initDesignVariables(cfg, dv);

//     for (int iter = 0; iter < cfg.opt_max_iter; ++iter)
//     {
//         OptimizationResult result = evaluateDesign(cfg, dv);

//         printOptimizationInfo(iter, dv, result);

//         updateDesignVariables(cfg, dv, result.grad);
//     }
// }
// OptimizationResult evaluateDesign(const SolverConfig& cfg,
//     const DesignVariables& dv)
// {
//     Mesh mesh0;
//     readMesh(cfg.meshFile, mesh0);

//     applyParameterization(mesh0, dv);

//     computeGeometry(mesh0);

//     dvmSolver primal(mesh0, cfg);
//     primal.run();

//     double J = computeObjective(primal, cfg.objective_type);

//     adjointDVM adj(primal);
//     adj.run();

//     std::vector<double> grad;
//     computeDesignGradient(mesh0, primal, adj, dv, grad);

//     OptimizationResult result;
//     result.J = J;
//     result.grad = grad;
//     return result;
// }