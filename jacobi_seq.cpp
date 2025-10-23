//g++ -O3 -march=native -std=c++17 jacobi_seq.cpp -o jacobi
//./jacobi --Nx 512 --Ny 512 --a 1 --b 1 --tol 1e-10 --quiet 1

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <optional>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

struct Params {
    int Nx = 256;         // interior points in x
    int Ny = 256;         // interior points in y
    double a = 1.0;       // domain length in x
    double b = 1.0;       // domain length in y
    int maxiter = 5'000'000;
    double tol = 1e-10;   // residual L2 tolerance
    int it_report = 100;  // print every k iterations
    int quiet = 0;        // 1 = suppress per-iteration logs
    int niters = -1;      // if >0, run exactly niters iterations
};

static inline size_t idx(int i, int j, int Nx, int Ny) {
    return static_cast<size_t>(j) * (Nx + 2) + static_cast<size_t>(i);
}

struct Field {
    int Nx, Ny; double dx, dy;
    std::vector<double> u, f; // size (Nx+2)*(Ny+2)
    Field(int Nx_, int Ny_, double a, double b) : Nx(Nx_), Ny(Ny_) {
        dx = a / (Nx + 1);
        dy = b / (Ny + 1);
        u.assign(static_cast<size_t>((Nx+2)*(Ny+2)), 0.0);
        f.assign(static_cast<size_t>((Nx+2)*(Ny+2)), 0.0);
    }
};

// Manufactured exact solution and RHS
struct Exact {
    double u(double x, double y, double a, double b) const {
        return std::sin(M_PI*x/a) * std::sin(M_PI*y/b);
    }
    double f(double x, double y, double a, double b) const {
        double coeff = - ( (M_PI*M_PI)/(a*a) + (M_PI*M_PI)/(b*b) );
        return coeff * u(x,y,a,b);
    }
};

void apply_ms_bc_and_rhs(Field &fld, const Params &p, const Exact &ex) {
    const int Nx=fld.Nx, Ny=fld.Ny; const double dx=fld.dx, dy=fld.dy;
    // boundaries from exact solution
    for (int i=0;i<=Nx+1;++i){ double x=i*dx; fld.u[idx(i,0,Nx,Ny)] = ex.u(x,0.0,p.a,p.b);
                                fld.u[idx(i,Ny+1,Nx,Ny)] = ex.u(x,p.b,p.a,p.b); }
    for (int j=0;j<=Ny+1;++j){ double y=j*dy; fld.u[idx(0,j,Nx,Ny)] = ex.u(0.0,y,p.a,p.b);
                                fld.u[idx(Nx+1,j,Nx,Ny)] = ex.u(p.a,y,p.a,p.b); }
    // interior RHS
    for (int j=1;j<=Ny;++j){ double y=j*dy; for (int i=1;i<=Nx;++i){ double x=i*dx; fld.f[idx(i,j,Nx,Ny)] = ex.f(x,y,p.a,p.b); }}
}

double compute_residual_L2(const Field &fld){
    const int Nx=fld.Nx, Ny=fld.Ny; const double dx=fld.dx, dy=fld.dy;
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    double sum=0.0;
    for (int j=1;j<=Ny;++j){
        for (int i=1;i<=Nx;++i){
            double uij = fld.u[idx(i,j,Nx,Ny)];
            double lap = (fld.u[idx(i+1,j,Nx,Ny)] - 2*uij + fld.u[idx(i-1,j,Nx,Ny)])*idx2
                       + (fld.u[idx(i,j+1,Nx,Ny)] - 2*uij + fld.u[idx(i,j-1,Nx,Ny)])*idy2;
            double r = lap - fld.f[idx(i,j,Nx,Ny)];
            sum += r*r;
        }
    }
    return std::sqrt(sum/(Nx*Ny));
}

void compute_errors(const Field &fld, const Params &p, const Exact &ex, double &L2, double &Linf){
    const int Nx=fld.Nx, Ny=fld.Ny; const double dx=fld.dx, dy=fld.dy;
    double sum=0.0, linf=0.0; size_t count=0;
    for (int j=0;j<=Ny+1;++j){ double y=j*dy; for (int i=0;i<=Nx+1;++i){ double x=i*dx;
        double e=std::fabs(fld.u[idx(i,j,Nx,Ny)] - ex.u(x,y,p.a,p.b)); sum+=e*e; linf=std::max(linf,e); ++count; }}
    L2 = std::sqrt(sum / static_cast<double>(count));
    Linf = linf;
}

int jacobi(Field &fld, const Params &p){
    // precomputation
    std::vector<double> u_new = fld.u;
    const int Nx=fld.Nx, Ny=fld.Ny; const double dx=fld.dx, dy=fld.dy;
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    const double denom = 2.0*(idx2+idy2);
    int it=0;

    if (p.niters > 0) {
        for (it=1; it<=p.niters; ++it){
            //sweep body
            for (int j=1;j<=Ny;++j){
                for (int i=1;i<=Nx;++i) {
                    double rhs = -fld.f[idx(i,j,Nx,Ny)]
                               + idx2*(fld.u[idx(i+1,j,Nx,Ny)] + fld.u[idx(i-1,j,Nx,Ny)])
                               + idy2*(fld.u[idx(i,j+1,Nx,Ny)] + fld.u[idx(i,j-1,Nx,Ny)]);
                    u_new[idx(i,j,Nx,Ny)] = rhs / denom;
                }
            }
            std::swap(fld.u, u_new);
            if (!p.quiet && (it % p.it_report == 0 || it == 1)) {
                double res = compute_residual_L2(fld);
                std::cout << "[Jacobi] it="<<it<<" resL2="<<std::scientific<<std::setprecision(6)<<res<<"\n";
            }
        }
        return it;
    } else {
        for (it=1; it<=p.maxiter; ++it){
            // sweeps body
            for (int j=1;j<=Ny;++j){
                for (int i=1;i<=Nx;++i) {
                    double rhs = -fld.f[idx(i,j,Nx,Ny)]
                               + idx2*(fld.u[idx(i+1,j,Nx,Ny)] + fld.u[idx(i-1,j,Nx,Ny)])
                               + idy2*(fld.u[idx(i,j+1,Nx,Ny)] + fld.u[idx(i,j-1,Nx,Ny)]);
                    u_new[idx(i,j,Nx,Ny)] = rhs / denom;
                }
            }
            std::swap(fld.u, u_new);
            double res = compute_residual_L2(fld);
            if (!p.quiet && (it % p.it_report == 0 || it == 1)) {
                std::cout << "[Jacobi] it="<<it<<" resL2="<<std::scientific<<std::setprecision(6)<<res<<"\n";
            }
            if (res < p.tol) break;
        }
        return it;
    }
}


void parse_cli(int argc, char** argv, Params &p){
auto get = [&](const std::string &key)->std::optional<std::string>{
for (int k=1;k<argc-1;++k){ if (std::string(argv[k])==key) return std::string(argv[k+1]); }
return std::nullopt;
};
if (auto v=get("--Nx")) p.Nx = std::stoi(*v);
if (auto v=get("--Ny")) p.Ny = std::stoi(*v);
if (auto v=get("--a")) p.a = std::stod(*v);
if (auto v=get("--b")) p.b = std::stod(*v);
if (auto v=get("--maxiter")) p.maxiter = std::stoi(*v);
if (auto v=get("--tol")) p.tol = std::stod(*v);
if (auto v=get("--it_report")) p.it_report = std::stoi(*v);
if (auto v=get("--quiet")) p.quiet = std::stoi(*v);
if (auto v=get("--niters")) p.niters = std::stoi(*v);
}

int main(int argc, char** argv){
    std::ios::sync_with_stdio(false);
    Params p; 
    parse_cli(argc, argv, p);

    Field fld(p.Nx, p.Ny, p.a, p.b);
    Exact ex;
    apply_ms_bc_and_rhs(fld, p, ex);

    //Time analysis
    auto t0 = std::chrono::high_resolution_clock::now();
    int it = jacobi(fld, p);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double L2=0.0, Linf=0.0; 
    compute_errors(fld,p,ex,L2,Linf);
    double res = compute_residual_L2(fld);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Iterations: " << it << "\n";
    std::cout << "Residual L2: " << res << "\n";
    std::cout << "Error   L2: " << L2  << "\n";
    std::cout << "Error Linf: " << Linf << "\n";
    std::cout << "Elapsed (s): " << std::scientific << std::setprecision(6) << elapsed << "\n";
    return 0;
}