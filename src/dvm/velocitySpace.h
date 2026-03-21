#ifndef VELOCITYSPACE_H
#define VELOCITYSPACE_H

#include "defs.h"
#include <array>
#include <cstddef>
#include <vector>

struct VelocitySpace {
    std::vector<double> vx, vy;
    std::vector<double> weight_x, weight_y;
    std::vector<double> Vx, Vy;
    std::vector<double> weight;
    std::vector<double> v2, feq;
    std::vector<std::array<double, Nmacro*2>> weight_macro;
    std::vector<std::array<double, Nmacro*2>> weight_coll;

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
