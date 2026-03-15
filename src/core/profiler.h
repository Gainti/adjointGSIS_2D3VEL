#ifndef PROFILER_H_INCLUDED
#define PROFILER_H_INCLUDED

#include <vector>
#include <string>
#include <cstdio>
#include <mpi.h>
#include <algorithm>
#include "halo.h"

struct HaloExchangeStats;

enum class DvmTimerID {
    StepTotal = 0,

    // top-level
    GetRhs,
    BoundarySet,
    UpdateMacroTotal,
    UpdateMacroAllreduce,

    // lusgs breakdown
    LusgsTotal,
    LusgsForwardTotal,
    LusgsBackwardTotal,

    LusgsForwardInterior,
    LusgsForwardBlockWait,
    LusgsBackwardInterior,
    LusgsBackwardBlockWait,

    // halo breakdown
    ExchangeTotal,
    ExchangePostRecv,
    ExchangePack,
    ExchangePostSend,
    ExchangeWait,
    ExchangeUnpack,

    Count
};

class DvmProfiler {
public:
    DvmProfiler() {
        const int n = static_cast<int>(DvmTimerID::Count);
        elapsed.assign(n, 0.0);
        calls.assign(n, 0);
        names.resize(n);

        names[(int)DvmTimerID::StepTotal]            = "step_total";

        names[(int)DvmTimerID::GetRhs]               = "getRhs";
        names[(int)DvmTimerID::LusgsTotal]           = "lusgs_total";
        names[(int)DvmTimerID::BoundarySet]          = "boundarySet";
        names[(int)DvmTimerID::UpdateMacroTotal]     = "updateMacro_total";
        names[(int)DvmTimerID::UpdateMacroAllreduce] = "updateMacro_allreduce";

        names[(int)DvmTimerID::LusgsForwardTotal]     = "lusgs_forward_total";
        names[(int)DvmTimerID::LusgsBackwardTotal]    = "lusgs_backward_total";

        names[(int)DvmTimerID::LusgsForwardInterior]  = "lusgs_forward_interior";
        names[(int)DvmTimerID::LusgsForwardBlockWait] = "lusgs_forward_block_wait";
        names[(int)DvmTimerID::LusgsBackwardInterior] = "lusgs_backward_interior";
        names[(int)DvmTimerID::LusgsBackwardBlockWait]= "lusgs_backward_block_wait";

        names[(int)DvmTimerID::ExchangeTotal]         = "exchange_total";

        names[(int)DvmTimerID::ExchangeTotal]        = "exchange_total";
        names[(int)DvmTimerID::ExchangePostRecv]     = "exchange_post_recv";
        names[(int)DvmTimerID::ExchangePack]         = "exchange_pack";
        names[(int)DvmTimerID::ExchangePostSend]     = "exchange_post_send";
        names[(int)DvmTimerID::ExchangeWait]         = "exchange_wait";
        names[(int)DvmTimerID::ExchangeUnpack]       = "exchange_unpack";
    }

    inline double tic() const {
        return MPI_Wtime();
    }

    inline void add(DvmTimerID id, double dt) {
        elapsed[(int)id] += dt;
        calls[(int)id]   += 1;
    }

    inline void reset() {
        std::fill(elapsed.begin(), elapsed.end(), 0.0);
        std::fill(calls.begin(), calls.end(), 0);
    }

    void addHaloStats(const HaloExchangeStats& s){
        add(DvmTimerID::ExchangeTotal,    s.t_total);
        add(DvmTimerID::ExchangePostRecv, s.t_post_recv);
        add(DvmTimerID::ExchangePack,     s.t_pack);
        add(DvmTimerID::ExchangePostSend, s.t_post_send);
        add(DvmTimerID::ExchangeWait,     s.t_wait);
        add(DvmTimerID::ExchangeUnpack,   s.t_unpack);
    }

    void report(MPI_Comm comm, int rank, const char* title = "[DVM profile]") const {
        const int n = static_cast<int>(DvmTimerID::Count);

        std::vector<double> t_sum(n, 0.0), t_max(n, 0.0), t_min(n, 0.0);
        std::vector<int>    c_sum(n, 0);

        MPI_Reduce(elapsed.data(), t_sum.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(elapsed.data(), t_max.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(elapsed.data(), t_min.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(calls.data(),   c_sum.data(), n, MPI_INT,    MPI_SUM, 0, comm);

        int size = 1;
        MPI_Comm_size(comm, &size);

        if (rank != 0) return;

        auto avg = [&](DvmTimerID id) { return t_sum[(int)id] / size; };
        auto mx  = [&](DvmTimerID id) { return t_max[(int)id]; };
        auto mn  = [&](DvmTimerID id) { return t_min[(int)id]; };
        auto cnt = [&](DvmTimerID id) { return static_cast<double>(c_sum[(int)id]) / size; };

        std::printf("\n%s\n", title);
        std::printf("%-30s %12s %12s %12s %12s %12s\n",
                    "name", "avg(s)", "max(s)", "min(s)", "calls(avg)", "max/share");

        double t_ref = mx(DvmTimerID::StepTotal);
        if (t_ref <= 0.0) t_ref = 1.0;

        auto print_one = [&](DvmTimerID id, int indent = 0) {
            std::string label(indent, ' ');
            label += names[(int)id];
            double share = 100.0 * mx(id) / t_ref;

            std::printf("%-30s %12.3e %12.3e %12.3e %12.2f %11.2f%%\n",
                        label.c_str(),
                        avg(id), mx(id), mn(id),
                        cnt(id), share);
        };

        // top-level
        print_one(DvmTimerID::StepTotal);
        print_one(DvmTimerID::GetRhs, 2);
        print_one(DvmTimerID::BoundarySet, 2);
        print_one(DvmTimerID::UpdateMacroTotal, 2);
        print_one(DvmTimerID::UpdateMacroAllreduce, 4);

        // lusgs detail
        print_one(DvmTimerID::LusgsTotal, 2);
        print_one(DvmTimerID::LusgsForwardTotal, 4);
        print_one(DvmTimerID::LusgsForwardInterior, 6);
        print_one(DvmTimerID::LusgsForwardBlockWait, 6);
        print_one(DvmTimerID::LusgsBackwardTotal, 4);
        print_one(DvmTimerID::LusgsBackwardInterior, 6);
        print_one(DvmTimerID::LusgsBackwardBlockWait, 6);

        // halo detail
        print_one(DvmTimerID::ExchangeTotal, 2);
        print_one(DvmTimerID::ExchangePostRecv, 4);
        print_one(DvmTimerID::ExchangePack, 4);
        print_one(DvmTimerID::ExchangePostSend, 4);
        print_one(DvmTimerID::ExchangeWait, 4);
        print_one(DvmTimerID::ExchangeUnpack, 4);

        std::printf("\n");
    }

private:
    std::vector<double> elapsed;
    std::vector<int> calls;
    std::vector<std::string> names;
};

class ScopedTimer {
public:
    ScopedTimer(DvmProfiler& profiler, DvmTimerID id)
        : profiler_(profiler), id_(id), t0_(MPI_Wtime()) {}

    ~ScopedTimer() {
        profiler_.add(id_, MPI_Wtime() - t0_);
    }

private:
    DvmProfiler& profiler_;
    DvmTimerID id_;
    double t0_;
};

#endif