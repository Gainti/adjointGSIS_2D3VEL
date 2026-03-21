#pragma once

constexpr int dim = 2;
constexpr int Nvdf = 2;
constexpr int Nmacro = 6; // rho,ux,uy,tau,qx,qy
constexpr int Namacro = 6; // rho, aux, auy, atau, aqx, aqy

struct SolverConfig {
    // constant
    int uwall=0;// 0: shear ; 1: x direction ; 2: y direction
    double tauw=0.0;
    double delta=0.0;
    double gamma=0.0;
    double Pr=0.0;
    double St=0.0;
    // velocity grid
    int  Nvx, Nvy, Nv;
    double Lvx, Lvy;
    // iteration 
    int max_iter = 20000;
    double tol = 1e-5;
    int print_interval = 10;
    int check_interval = 1;
};

// Enum for boundary condition types, based on Fluent's documentation
enum class BCType {
    internal = 2,
    wall =3,
    pressure_inlet = 4,
    pressure_outlet = 5,
    symmetry = 7,
    periodic_shadow = 8,
    pressure_far_field = 9,
    velocity_inlet = 10,
    periodic = 12,
    fan = 14,
    mass_flow_inlet = 20,
    interface = 24,
    parent = 31,
    outflow = 36,
    axis = 37
};

// typedef std::complex<double> scalar;
typedef double scalar;
// constexpr scalar Zero = scalar(0.0, 0.0);
constexpr scalar Zero = scalar(0.0);

#define PI 3.1415926535897932
#define PI2 1.5707963267948966
#define inv_2PI 0.159154943091895
#define SQRT2divPI 0.797884560802865
#define RAD 1.7453292519943296e-2
#define funPI 0.179587122125167 // PI^(-1.5)
#define sqrtPI 1.772453850905516