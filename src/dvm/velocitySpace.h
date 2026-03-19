#ifndef VELOCITYSPACE_H
#define VELOCITYSPACE_H

#include "defs.h"
#include <array>
#include <cstddef>
#include <vector>

struct VelocitySpace {
    std::vector<double> vx, vy, vz;
    std::vector<double> weight_x, weight_y, weight_z;
    std::vector<double> Vx, Vy, Vz;
    std::vector<double> weight;
    std::vector<double> c2, feq, exp_c2;
    std::vector<std::array<double, Nmacro>> weight_macro;

    bool build(const SolverConfig& cfg);

private:
    static void uniform(std::vector<double>& vi,
                        std::vector<double>& weight_i,
                        size_t Nvi,
                        double Lvi);

    static void nonUniform(std::vector<double>& vi,
                           std::vector<double>& weight_i,
                           size_t Nvi,
                           double Lvi);
};

#endif // VELOCITYSPACE_H
