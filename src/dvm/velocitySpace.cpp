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
    uniform(vz, weight_z, cfg.Nvz, cfg.Lvz);

    if (cfg.Nv != static_cast<int>(vx.size() * vy.size() * vz.size())) {
        printf("Error: Inconsistent velocity space dimensions\\n\\n");
        return false;
    }

    Vx.resize(cfg.Nv);
    Vy.resize(cfg.Nv);
    Vz.resize(cfg.Nv);
    weight.resize(cfg.Nv);
    c2.resize(cfg.Nv);
    feq.resize(cfg.Nv);
    exp_c2.resize(cfg.Nv);

    size_t ip = 0;
    for (size_t ipz = 0; ipz < vz.size(); ipz++) {
        for (size_t ipy = 0; ipy < vy.size(); ipy++) {
            for (size_t ipx = 0; ipx < vx.size(); ipx++) {
                Vx[ip] = vx[ipx];
                Vy[ip] = vy[ipy];
                Vz[ip] = vz[ipz];
                weight[ip] = weight_x[ipx] * weight_y[ipy] * weight_z[ipz];
                ip++;
            }
        }
    }

    for (int vi = 0; vi < cfg.Nv; vi++) {
        c2[vi] = Vx[vi] * Vx[vi] + Vy[vi] * Vy[vi] + Vz[vi] * Vz[vi];
        feq[vi] = std::exp(-c2[vi]) / PI / sqrtPI;
        exp_c2[vi] = std::exp(-c2[vi]);
    }

    weight_macro.resize(cfg.Nv);
    for (int vi = 0; vi < cfg.Nv; vi++) {
        double fw = feq[vi] * weight[vi];
        weight_macro[vi][0] = fw;
        weight_macro[vi][1] = fw * Vx[vi];
        weight_macro[vi][2] = fw * Vy[vi];
        weight_macro[vi][3] = fw * (2.0 * c2[vi] / 3.0 - 1.0);
        weight_macro[vi][4] = fw * (c2[vi] - 2.5) * Vx[vi];
        weight_macro[vi][5] = fw * (c2[vi] - 2.5) * Vy[vi];
        weight_macro[vi][6] = fw * 2.0 * (Vx[vi] * Vx[vi] - c2[vi] / 3.0);
        weight_macro[vi][7] = fw * 2.0 * Vx[vi] * Vy[vi];
        weight_macro[vi][8] = fw * 2.0 * Vx[vi] * Vy[vi];
        weight_macro[vi][9] = fw * 2.0 * (Vy[vi] * Vy[vi] - c2[vi] / 3.0);
    }

    return true;
}
