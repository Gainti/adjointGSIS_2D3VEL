#include "app.h"

#include "utils.h"
#include "config.h"

static void usage() {
    printf("Usage:\n");
    printf("  ./solver --case <caseDir>\n\n");
}

void printConfig(const SolverConfig& cfg, int rank) {
    if (rank != 0) return;

    printf("\n========== Solver Config ==========\n");

    printf("[wall / physical]\n");
    printf("  uwall           = %d\n", cfg.uwall);
    printf("  tauw            = %.12g\n", cfg.tauw);
    printf("  delta           = %.12g\n", cfg.delta);
    printf("  gamma           = %.12g\n", cfg.gamma);
    printf("  Pr              = %.12g\n", cfg.Pr);
    printf("  St              = %.12g\n", cfg.St);

    printf("\n[velocity space]\n");
    printf("  Nvx             = %d\n", cfg.Nvx);
    printf("  Nvy             = %d\n", cfg.Nvy);
    printf("  Nvz             = %d\n", cfg.Nvz);
    printf("  Nv              = %d\n", cfg.Nv);
    printf("  Lvx             = %.12g\n", cfg.Lvx);
    printf("  Lvy             = %.12g\n", cfg.Lvy);
    printf("  Lvz             = %.12g\n", cfg.Lvz);

    printf("\n[iteration]\n");
    printf("  max_iter        = %d\n", cfg.max_iter);
    printf("  tol             = %.12g\n", cfg.tol);
    printf("  print_interval  = %d\n", cfg.print_interval);
    printf("  check_interval  = %d\n", cfg.check_interval);

    printf("===================================\n\n");
}

bool readCaseConfig(int argc, char** argv,
    CaseInfo& case_info,
    SolverConfig& scfg,
    int rank) {
    for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--case" && i + 1 < argc) {
            case_info.case_dir = argv[i + 1];
            case_info.config_path = utils::join_path(case_info.case_dir, "config.ini");
            break;
        }
    }

    if (case_info.config_path.empty()) {
        if (rank == 0) {
            usage();
        }
        return false;
    }

    Config cfg;
    if (!cfg.load(case_info.config_path)) {
    if (rank == 0) {
            printf("Failed to read config: %s\n\n", case_info.config_path.c_str());
        }
        return false;
    }

    std::string base_dir = utils::dirname(case_info.config_path);
    case_info.mesh_file  = cfg.get_string("case", "mesh_file", "mesh");
    case_info.output_dir = cfg.get_string("case", "output_dir", "output");

    if (case_info.mesh_file.empty()) {
        if (rank == 0) {
            printf("Config missing case.mesh_file\n\n");
        }
        return false;
    }

    case_info.mesh_file  = utils::join_path(base_dir, case_info.mesh_file);
    case_info.output_dir = utils::join_path(base_dir, case_info.output_dir);

    scfg.uwall = cfg.get_int("solver", "uwall", 0);
    scfg.tauw  = cfg.get_double("solver", "tauw", 0.0);
    scfg.delta = cfg.get_double("solver", "delta", 0.0);
    scfg.gamma = cfg.get_double("solver", "gamma", 5.0 / 3.0);
    scfg.Pr    = cfg.get_double("solver", "Pr", 2.0 / 3.0);
    scfg.St    = cfg.get_double("solver", "St", 0.0);

    scfg.Nvx = cfg.get_int("solver", "Nvx", 0);
    scfg.Nvy = cfg.get_int("solver", "Nvy", 0);
    scfg.Nvz = cfg.get_int("solver", "Nvz", 0);
    scfg.Nv  = scfg.Nvx * scfg.Nvy * scfg.Nvz;

    scfg.Lvx = cfg.get_double("solver", "Lvx", 0.0);
    scfg.Lvy = cfg.get_double("solver", "Lvy", 0.0);
    scfg.Lvz = cfg.get_double("solver", "Lvz", 0.0);

    scfg.max_iter       = cfg.get_int("solver", "maxIter", 2000);
    scfg.tol            = cfg.get_double("solver", "tol", 1e-5);
    scfg.print_interval = cfg.get_int("solver", "printInterval", 10);
    scfg.check_interval = cfg.get_int("solver", "checkInterval", 1);

    printConfig(scfg, rank);
    return true;
}