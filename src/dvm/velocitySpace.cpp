#include "velocitySpace.h"

#include <cmath>
#include <cstdio>

void VelocitySpace::uniform(std::vector<double>& vi,
                            std::vector<double>& weight_i,
                            size_t Nvi,
                            double Lvi)
{
    vi.resize(Nvi);
    weight_i.resize(Nvi);

    double dx = 2.0 * Lvi / static_cast<double>(Nvi - 1);
    for (size_t ipx = 0; ipx < Nvi; ipx++) {
        vi[ipx] = -Lvi + dx * ipx;
        weight_i[ipx] = dx;
    }
}

void VelocitySpace::nonUniform(std::vector<double>& vi,
                               std::vector<double>& weight_i,
                               size_t Nvi,
                               double Lvi)
{
    vi.resize(Nvi);
    weight_i.resize(Nvi);

    for (size_t ipx = 0; ipx < Nvi; ipx++) {
        int ix = 2 * ipx - (Nvi - 1);
        vi[ipx] = Lvi * std::pow(ix, 3) / std::pow(Nvi - 1, 3);
        weight_i[ipx] = 6.0 * Lvi * std::pow(ix, 2) / std::pow(Nvi - 1, 3);
    }
}

bool VelocitySpace::build(const SolverConfig& cfg)
{
    nonUniform(vx, weight_x, cfg.Nvx, cfg.Lvx);
    nonUniform(vy, weight_y, cfg.Nvy, cfg.Lvy);

    if (cfg.Nv != static_cast<int>(vx.size() * vy.size())) {
        printf("Error: Inconsistent velocity space dimensions\\n\\n");
        return false;
    }

    Vx.resize(cfg.Nv);
    Vy.resize(cfg.Nv);
    weight.resize(cfg.Nv);
    v2.resize(cfg.Nv);
    feq.resize(cfg.Nv);

    size_t ip = 0;
    for (size_t ipy = 0; ipy < vy.size(); ipy++) {
        for (size_t ipx = 0; ipx < vx.size(); ipx++) {
            Vx[ip] = vx[ipx];
            Vy[ip] = vy[ipy];
            weight[ip] = weight_x[ipx] * weight_y[ipy];
            ip++;
        }
    }

    for (int vi = 0; vi < cfg.Nv; vi++) {
        v2[vi] = Vx[vi] * Vx[vi] + Vy[vi] * Vy[vi];
        feq[vi] = std::exp(-v2[vi]) / PI;
    }

    weight_macro.resize(cfg.Nv);
    weight_coll.resize(cfg.Nv);
    for (int vi = 0; vi < cfg.Nv; vi++) {
        // macro
        weight_macro[vi][0] = 1.0;
        weight_macro[vi][1] = 0.0;
        weight_macro[vi][2] = Vx[vi];
        weight_macro[vi][3] = 0.0;
        weight_macro[vi][4] = Vy[vi];
        weight_macro[vi][5] = 0.0;
        weight_macro[vi][6] = 2.0 * v2[vi] / 3.0 - 1.0;
        weight_macro[vi][7] = 1.0/3.0;
        weight_macro[vi][8] = (v2[vi] - 2.5) * Vx[vi];
        weight_macro[vi][9] = 0.5 * Vx[vi];
        weight_macro[vi][10] = (v2[vi] - 2.5) * Vy[vi];
        weight_macro[vi][11] = 0.5 * Vy[vi];
        // collision
        weight_coll[vi][0] = 1.0;
        weight_coll[vi][1] = 1.0;
        weight_coll[vi][2] = 2.0 * Vx[vi];
        weight_coll[vi][3] = 2.0 * Vx[vi];
        weight_coll[vi][4] = 2.0*Vy[vi];
        weight_coll[vi][5] = 2.0*Vy[vi];
        weight_coll[vi][6] = v2[vi] - 1.0;
        weight_coll[vi][7] = v2[vi];
        weight_coll[vi][8] = 4.0/15.0*(v2[vi] - 2.0)*Vx[vi];
        weight_coll[vi][9] = 4.0/15.0*(v2[vi] - 1.0)*Vx[vi];
        weight_coll[vi][10] = 4.0/15.0*(v2[vi] - 2.0)*Vy[vi];
        weight_coll[vi][11] = 4.0/15.0*(v2[vi] - 1.0)*Vy[vi];
    }

    return true;
}
