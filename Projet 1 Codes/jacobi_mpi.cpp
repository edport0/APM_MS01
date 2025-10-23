// jacobi_mpi.cpp 
//
// PDE:   Δu = f,   with f(x,y) = 0 in Ω=(0,a)×(0,b)
// BCs:   u(x,0)=U0,  u(x,b)=U0,  u(a,y)=U0,  u(0,y)=U0*(1+α V(y)),
//        V(y)=1 - cos(2π y / b),   0<α<1.
//
//
// Compilation:  mpicxx -O3 -std=c++17 jacobi_mpi.cpp -o jacobi_mpi
// Run:    mpirun -np 4 ./jacobi_mpi --Nx 1024 --Ny 1024 --a 1 --b 1 
//            --U0 1.0 --alpha 0.2 --niters 100000 --quiet 1

#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Params {
    int Nx=1024, Ny=2048;        // global interior sizes
    double a=1.0, b=1.0;       // domain lengths
    double U0=1.0;             // base boundary value
    double alpha=0.2;          // amplitude for left boundary modulation (|alpha|<1)
    int maxiter=2000000;       // residual stop cap
    double tol=1e-10;          // residual tolerance (global L2)
    int it_report=100;         // print frequency
    int quiet=1;               // suppress periodic logs
    int niters=100000;             // if >0, do exactly niters iterations (for timing)
};

static void parse_cli(int argc, char** argv, Params &p){
    auto get=[&](const std::string& k)->const char*{ for(int i=1;i<argc-1;++i) if(k==argv[i]) return argv[i+1]; return nullptr; };
    if (const char* v=get("--Nx")) p.Nx=std::stoi(v);
    if (const char* v=get("--Ny")) p.Ny=std::stoi(v);
    if (const char* v=get("--a"))  p.a =std::stod(v);
    if (const char* v=get("--b"))  p.b =std::stod(v);
    if (const char* v=get("--U0")) p.U0=std::stod(v);
    if (const char* v=get("--alpha")) p.alpha=std::stod(v);
    if (const char* v=get("--maxiter")) p.maxiter=std::stoi(v);
    if (const char* v=get("--tol"))     p.tol    =std::stod(v);
    if (const char* v=get("--it_report")) p.it_report=std::stoi(v);
    if (const char* v=get("--quiet")) p.quiet=std::stoi(v);
    if (const char* v=get("--niters")) p.niters=std::stoi(v);
}

// Local linear index for array of size (nxloc+2) x (Ny+2)
static inline size_t ID(int i,int j,int nxloc,int Ny){ return (size_t)j*(nxloc+2)+(size_t)i; }

static inline double Vfun(double y, double b){ return 1.0 - std::cos(2.0*M_PI*y/b); }

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    Params p; parse_cli(argc,argv,p);

    // 1) 1D x-slab decomposition
    std::vector<int> nx(size, p.Nx/size); int rem=p.Nx%size; for(int r=0;r<rem;++r) nx[r]++;
    std::vector<int> off(size,0); for(int r=1;r<size;++r) off[r]=off[r-1]+nx[r-1];
    const int nxloc=nx[rank]; const int xoff=off[rank];

    // 2) Geometry & coefficients
    const double dx=p.a/(p.Nx+1), dy=p.b/(p.Ny+1);
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), denom=2.0*(idx2+idy2);

    // 3) Allocate local arrays (with halos)
    std::vector<double> u((nxloc+2)*(p.Ny+2), 0.0), unew(u.size(),0.0);
    // f(x,y)=0 everywhere inside

    auto X=[&](int iL){ int ig = xoff + iL; return ig*dx; };
    auto Y=[&](int j){ return j*dy; };

    // 4) Apply Dirichlet BCs 
    // Top/bottom boundaries on all ranks
    for(int i=0;i<=nxloc+1;++i){
        u[ID(i,0,      nxloc,p.Ny)] = p.U0;           // y=0
        u[ID(i,p.Ny+1, nxloc,p.Ny)] = p.U0;           // y=b
    }
    // Left boundary only on global left (rank 0)
    if(rank==0){
        for(int j=0;j<=p.Ny+1;++j){ double y=Y(j);
            u[ID(0,j,nxloc,p.Ny)] = p.U0 * (1.0 + p.alpha * Vfun(y,p.b));
        }
    }
    // Right boundary only on global right (last rank)
    if(rank==size-1){
        for(int j=0;j<=p.Ny+1;++j){ u[ID(nxloc+1,j,nxloc,p.Ny)] = p.U0; }
    }

    // 5) Halo buffers (columns)
    std::vector<double> sendL(p.Ny+2), sendR(p.Ny+2), recvL(p.Ny+2), recvR(p.Ny+2);
    auto halo=[&](){
        for(int j=0;j<=p.Ny+1;++j){ sendL[j]=u[ID(1,j,nxloc,p.Ny)]; sendR[j]=u[ID(nxloc,j,nxloc,p.Ny)]; }
        int L=(rank==0?-1:rank-1), R=(rank==size-1?-1:rank+1);
        MPI_Request req[4]; int q=0;
        if(L!=-1) MPI_Irecv(recvL.data(), p.Ny+2, MPI_DOUBLE, L, 11, MPI_COMM_WORLD, &req[q++]);
        if(R!=-1) MPI_Irecv(recvR.data(), p.Ny+2, MPI_DOUBLE, R, 22, MPI_COMM_WORLD, &req[q++]);
        if(L!=-1) MPI_Isend(sendL.data(), p.Ny+2, MPI_DOUBLE, L, 22, MPI_COMM_WORLD, &req[q++]);
        if(R!=-1) MPI_Isend(sendR.data(), p.Ny+2, MPI_DOUBLE, R, 11, MPI_COMM_WORLD, &req[q++]);
        if(q) MPI_Waitall(q, req, MPI_STATUSES_IGNORE);
        if(L!=-1) for(int j=0;j<=p.Ny+1;++j) u[ID(0,       j,nxloc,p.Ny)] = recvL[j];
        if(R!=-1) for(int j=0;j<=p.Ny+1;++j) u[ID(nxloc+1, j,nxloc,p.Ny)] = recvR[j];
    };

    auto sweep=[&](){
        // Jacobi update with f=0
        for(int j=1;j<=p.Ny;++j){
            for(int i=1;i<=nxloc;++i){
                double rhs = idx2*(u[ID(i+1,j,nxloc,p.Ny)] + u[ID(i-1,j,nxloc,p.Ny)])
                           + idy2*(u[ID(i,  j+1,nxloc,p.Ny)] + u[ID(i,  j-1,nxloc,p.Ny)]);
                unew[ID(i,j,nxloc,p.Ny)] = rhs/denom;
            }
        }
        // Keep boundaries/halos consistent
        for(int j=0;j<=p.Ny+1;++j){ unew[ID(0,j,nxloc,p.Ny)]=u[ID(0,j,nxloc,p.Ny)]; unew[ID(nxloc+1,j,nxloc,p.Ny)]=u[ID(nxloc+1,j,nxloc,p.Ny)]; }
        for(int i=0;i<=nxloc+1;++i){ unew[ID(i,0,nxloc,p.Ny)]=u[ID(i,0,nxloc,p.Ny)]; unew[ID(i,p.Ny+1,nxloc,p.Ny)]=u[ID(i,p.Ny+1,nxloc,p.Ny)]; }
        u.swap(unew);
    };

    auto resid_local=[&](){
        double s=0.0; // f=0 so r = Δ_h u
        for(int j=1;j<=p.Ny;++j){
            for(int i=1;i<=nxloc;++i){
                double uij=u[ID(i,j,nxloc,p.Ny)];
                double lap=(u[ID(i+1,j,nxloc,p.Ny)]-2*uij+u[ID(i-1,j,nxloc,p.Ny)])*idx2
                         +(u[ID(i,  j+1,nxloc,p.Ny)]-2*uij+u[ID(i,  j-1,nxloc,p.Ny)])*idy2;
                s += lap*lap;
            }
        }
        return s;
    };

    // --- iterations & timing ---
    int it = 0;
    double resL2 = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (p.niters > 0) {
        for (it = 1; it <= p.niters; ++it) {
            halo();
            sweep();
            if (!p.quiet && (it % p.it_report == 0 || it == 1)) {
                double loc = resid_local(), glob = 0.0;
                MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                if (rank == 0) {
                    double r = std::sqrt(glob / (static_cast<double>(p.Nx) * p.Ny));
                    std::cout << "[it=" << it << "] resL2=" << std::scientific
                              << std::setprecision(6) << r << "\n";
                }
            }
        }
        double loc = resid_local(), glob = 0.0;
        MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        resL2 = std::sqrt(glob / (static_cast<double>(p.Nx) * p.Ny));
    } else {
        for (it = 1; it <= p.maxiter; ++it) {
            halo();
            sweep();
            double loc = resid_local(), glob = 0.0;
            MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            resL2 = std::sqrt(glob / (static_cast<double>(p.Nx) * p.Ny));
            if (!p.quiet && (it % p.it_report == 0 || it == 1)) {
                if (rank == 0)
                    std::cout << "[it=" << it << "] resL2=" << std::scientific
                              << std::setprecision(6) << resL2 << "\n";
            }
            if (resL2 < p.tol) break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Iterations: " << it << "\n";
        std::cout << "Residual L2: " << resL2 << "\n";
        std::cout << "Elapsed (s): " << (t1 - t0) << "\n";
    }

    MPI_Finalize();
    return 0;
}
