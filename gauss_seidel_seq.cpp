// g++ -O3 -march=native -std=c++17 gauss_seidel_seq.cpp -o gs_seq
// ./gs_seq --Nx 256 --Ny 256 --a 1 --b 1 --tol 1e-10 --it_report 200
// ./gs_seq --Nx 512 --Ny 512 --a 1 --b 1 --niters 10000 --quiet 1

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

struct Params {
    int Nx=128, Ny=128;
    double a=1.0, b=1.0;
    int maxiter=5'000'000;
    double tol=1e-10;
    int it_report=100;
    int quiet=0;
    int niters=-1;              // >0 => fixed number of iterations (benchmark mode)
};

static inline size_t ID(int i,int j,int Nx,int Ny){ return (size_t)j*(Nx+2)+(size_t)i; }

// Manufactured exact solution and RHS
static inline double u_exact(double x,double y,double a,double b){
    return std::sin(M_PI*x/a)*std::sin(M_PI*y/b);
}
static inline double f_exact(double x,double y,double a,double b){
    double c=-( (M_PI*M_PI)/(a*a) + (M_PI*M_PI)/(b*b) );
    return c*u_exact(x,y,a,b);
}

static void parse_cli(int argc, char** argv, Params &p){
    auto get=[&](const std::string& k)->const char*{
        for(int i=1;i<argc-1;++i) if(k==argv[i]) return argv[i+1];
        return nullptr;
    };
    if (const char* v=get("--Nx")) p.Nx=std::stoi(v);
    if (const char* v=get("--Ny")) p.Ny=std::stoi(v);
    if (const char* v=get("--a"))  p.a =std::stod(v);
    if (const char* v=get("--b"))  p.b =std::stod(v);
    if (const char* v=get("--maxiter")) p.maxiter=std::stoi(v);
    if (const char* v=get("--tol"))     p.tol    =std::stod(v);
    if (const char* v=get("--it_report")) p.it_report=std::stoi(v);
    if (const char* v=get("--quiet")) p.quiet=std::stoi(v);
    if (const char* v=get("--niters")) p.niters=std::stoi(v);
}

static double residual_L2(const std::vector<double>& u,
                          const std::vector<double>& f,
                          int Nx,int Ny,double dx,double dy){
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    double sum=0.0;
    for(int j=1;j<=Ny;++j){
        for(int i=1;i<=Nx;++i){
            double uij=u[ID(i,j,Nx,Ny)];
            double lap=(u[ID(i+1,j,Nx,Ny)]-2*uij+u[ID(i-1,j,Nx,Ny)])*idx2
                      +(u[ID(i,j+1,Nx,Ny)]-2*uij+u[ID(i,j-1,Nx,Ny)])*idy2;
            double r=lap - f[ID(i,j,Nx,Ny)];
            sum+=r*r;
        }
    }
    return std::sqrt(sum/(Nx*Ny));
}

static void compute_errors(const std::vector<double>& u,
                           int Nx,int Ny,double a,double b,double dx,double dy,
                           double &L2, double &Linf){
    double sum=0.0, linf=0.0; size_t count=0;
    for(int j=0;j<=Ny+1;++j){
        double y=j*dy;
        for(int i=0;i<=Nx+1;++i){
            double x=i*dx;
            double e = std::fabs(u[ID(i,j,Nx,Ny)] - u_exact(x,y,a,b));
            sum += e*e; linf = std::max(linf, e); ++count;
        }
    }
    L2 = std::sqrt(sum / static_cast<double>(count));
    Linf = linf;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    Params p; parse_cli(argc,argv,p);

    const double dx=p.a/(p.Nx+1), dy=p.b/(p.Ny+1);
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy);
    const double denom=2.0*(idx2+idy2);

    std::vector<double> u((p.Nx+2)*(p.Ny+2),0.0), f((p.Nx+2)*(p.Ny+2),0.0);

    // Dirichlet BCs from exact solution
    for(int i=0;i<=p.Nx+1;++i){
        double x=i*dx;
        u[ID(i,0,        p.Nx,p.Ny)] = u_exact(x,0,  p.a,p.b);
        u[ID(i,p.Ny+1,   p.Nx,p.Ny)] = u_exact(x,p.b,p.a,p.b);
    }
    for(int j=0;j<=p.Ny+1;++j){
        double y=j*dy;
        u[ID(0,       j, p.Nx,p.Ny)] = u_exact(0,   y,p.a,p.b);
        u[ID(p.Nx+1,  j, p.Nx,p.Ny)] = u_exact(p.a, y,p.a,p.b);
    }
    // RHS inside
    for(int j=1;j<=p.Ny;++j){
        double y=j*dy;
        for(int i=1;i<=p.Nx;++i){
            double x=i*dx;
            f[ID(i,j,p.Nx,p.Ny)] = f_exact(x,y,p.a,p.b);
        }
    }

    int it=0;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (p.niters > 0){
        for(it=1; it<=p.niters; ++it){
            for(int j=1;j<=p.Ny;++j){
                for(int i=1;i<=p.Nx;++i){
                    double rhs = -f[ID(i,j,p.Nx,p.Ny)]
                               + idx2*(u[ID(i+1,j,p.Nx,p.Ny)] + u[ID(i-1,j,p.Nx,p.Ny)])
                               + idy2*(u[ID(i,  j+1,p.Nx,p.Ny)] + u[ID(i,  j-1,p.Nx,p.Ny)]);
                    u[ID(i,j,p.Nx,p.Ny)] = rhs/denom;
                }
            }
            if(!p.quiet && (it%p.it_report==0 || it==1)){
                double res = residual_L2(u,f,p.Nx,p.Ny,dx,dy);
                std::cout<<"[GS] it="<<it<<" resL2="<<std::scientific<<std::setprecision(6)<<res<<"\n";
            }
        }
    } else {
        for(it=1; it<=p.maxiter; ++it){
            for(int j=1;j<=p.Ny;++j){
                for(int i=1;i<=p.Nx;++i){
                    double rhs = -f[ID(i,j,p.Nx,p.Ny)]
                               + idx2*(u[ID(i+1,j,p.Nx,p.Ny)] + u[ID(i-1,j,p.Nx,p.Ny)])
                               + idy2*(u[ID(i,  j+1,p.Nx,p.Ny)] + u[ID(i,  j-1,p.Nx,p.Ny)]);
                    u[ID(i,j,p.Nx,p.Ny)] = rhs/denom;
                }
            }
            double res = residual_L2(u,f,p.Nx,p.Ny,dx,dy);
            if(!p.quiet && (it%p.it_report==0 || it==1)){
                std::cout<<"[GS] it="<<it<<" resL2="<<std::scientific<<std::setprecision(6)<<res<<"\n";
            }
            if(res < p.tol) break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Final metrics
    double res = residual_L2(u,f,p.Nx,p.Ny,dx,dy);
    double L2=0.0, Linf=0.0;
    compute_errors(u, p.Nx, p.Ny, p.a, p.b, dx, dy, L2, Linf);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Iterations: " << it        << "\n";
    std::cout << "Residual L2: " << res      << "\n";
    std::cout << "Error   L2: " << L2        << "\n";
    std::cout << "Error Linf: " << Linf      << "\n";
    std::cout << "Elapsed (s): " << elapsed  << "\n";
    return 0;
}
