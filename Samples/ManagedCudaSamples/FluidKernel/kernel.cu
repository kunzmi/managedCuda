//Includes for IntelliSense

#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h> 
#include "float.h"
#include <builtin_types.h> 
#include <vector_functions.h> 


// Texture reference for reading velocity field
texture<float2, 2> texref;

extern "C" 
{
typedef unsigned long DWORD;

typedef struct vertex {
    float x, y, z;
    DWORD c;     
} Vertex;

// Note that these kernels are designed to work with arbitrary 
// domain sizes, not just domains that are multiples of the tile
// size. Therefore, we have extra code that checks to make sure
// a given thread location falls within the domain boundaries in
// both X and Y. Also, the domain is covered by looping over
// multiple elements in the Y direction, while there is a one-to-one
// mapping between threads in X and the tile size in X.
// Nolan Goodnight 9/22/06

// This method adds constant force vectors to the velocity field 
// stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
__global__ void 
addForces_k(float2 *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float2 *fj = (float2*)((char*)v + (ty + spy) * pitch) + tx + spx;

    float2 vterm = *fj;
    tx -= r; ty -= r;
    float s = 1.f / (1.f + tx*tx*tx*tx + ty*ty*ty*ty);
    vterm.x += s * fx;
    vterm.y += s * fy;
    *fj = vterm;
}

// This method performs the velocity advection step, where we
// trace velocity vectors back in time to update each grid cell.
// That is, v(x,t+1) = v(p(x,-dt),t). Here we perform bilinear
// interpolation in the velocity space.
__global__ void 
advectVelocity_k(float2 *v, float *vx, float *vy,
                 int dx, int pdx, int dy, float dt, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float2 vterm, ploc;
    float vxterm, vyterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * pdx + gtidx;
                vterm = tex2D(texref, (float)gtidx, (float)fi);
                ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
                ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
                vterm = tex2D(texref, ploc.x, ploc.y);
                vxterm = vterm.x; vyterm = vterm.y; 
                vx[fj] = vxterm;
                vy[fj] = vyterm; 
            }
        }
    }
}

// This method performs velocity diffusion and forces mass conservation 
// in the frequency domain. The inputs 'vx' and 'vy' are complex-valued 
// arrays holding the Fourier coefficients of the velocity field in
// X and Y. Diffusion in this space takes a simple form described as:
// v(k,t) = v(k,t) / (1 + visc * dt * k^2), where visc is the viscosity,
// and k is the wavenumber. The projection step forces the Fourier
// velocity vectors to be orthogonal to the vectors for each
// wavenumber: v(k,t) = v(k,t) - ((k dot v(k,t) * k) / k^2.
__global__ void 
diffuseProject_k(float2 *vx, float2 *vy, int dx, int dy, float dt, 
                 float visc, int lb) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float2 xterm, yterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                xterm = vx[fj];
                yterm = vy[fj];

                // Compute the index of the wavenumber based on the
                // data order produced by a standard NN FFT.
                int iix = gtidx;
                int iiy = (fi>dy/2)?(fi-(dy)):fi;

                // Velocity diffusion
                float kk = (float)(iix * iix + iiy * iiy); // k^2 
                float diff = 1.f / (1.f + visc * dt * kk);
                xterm.x *= diff; xterm.y *= diff;
                yterm.x *= diff; yterm.y *= diff;

                // Velocity projection
                if (kk > 0.f) {
                    float rkk = 1.f / kk;
                    // Real portion of velocity projection
                    float rkp = (iix * xterm.x + iiy * yterm.x);
                    // Imaginary portion of velocity projection
                    float ikp = (iix * xterm.y + iiy * yterm.y);
                    xterm.x -= rkk * rkp * iix;
                    xterm.y -= rkk * ikp * iix;
                    yterm.x -= rkk * rkp * iiy;
                    yterm.y -= rkk * ikp * iiy;
                }
                
                vx[fj] = xterm;
                vy[fj] = yterm;
            }
        }
    }
}

// This method updates the velocity field 'v' using the two complex 
// arrays from the previous step: 'vx' and 'vy'. Here we scale the 
// real components by 1/(dx*dy) to account for an unnormalized FFT. 
__global__ void 
updateVelocity_k(float2 *v, float *vx, float *vy, 
                 int dx, int pdx, int dy, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    float vxterm, vyterm;
    float2 nvterm;
    // gtidx is the domain location in x for this thread
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fjr = fi * pdx + gtidx; 
                vxterm = vx[fjr];
                vyterm = vy[fjr];

                // Normalize the result of the inverse FFT
                float scale = 1.f / (dx * dy);
                nvterm.x = vxterm * scale;
                nvterm.y = vyterm * scale;
               
                float2 *fj = (float2*)((char*)v + fi * pitch) + gtidx;
                *fj = nvterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}

// This method updates the particles by moving particle positions
// according to the velocity field and time step. That is, for each
// particle: p(t+1) = p(t) + dt * v(p(t)).  
__global__ void 
advectParticles_k(Vertex *part, float2 *v, int dx, int dy, 
                  float dt, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    float2 vterm;
    Vertex pterm;
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                pterm = part[fj];
                
                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((float2*)((char*)v + yvi * pitch) + xvi);   
 
                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;            
                pterm.x += 1.f; 
                pterm.x = pterm.x - (int)pterm.x;              
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;            
                pterm.y += 1.f; 
                pterm.y = pterm.y - (int)pterm.y;                  

                part[fj] = pterm;


            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}


// This method is the same as for DirectX, but vertex data type is different  
__global__ void 
advectParticles_OGL(float2 *part, float2 *v, int dx, int dy, 
                  float dt, int lb, size_t pitch) {

    int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
    int p;

    // gtidx is the domain location in x for this thread
    float2 pterm, vterm;
    if (gtidx < dx) {
        for (p = 0; p < lb; p++) {
            // fi is the domain location in y for this thread
            int fi = gtidy + p;
            if (fi < dy) {
                int fj = fi * dx + gtidx;
                pterm = part[fj];
                
                int xvi = ((int)(pterm.x * dx));
                int yvi = ((int)(pterm.y * dy));
                vterm = *((float2*)((char*)v + yvi * pitch) + xvi);   
 
                pterm.x += dt * vterm.x;
                pterm.x = pterm.x - (int)pterm.x;            
                pterm.x += 1.f; 
                pterm.x = pterm.x - (int)pterm.x;              
                pterm.y += dt * vterm.y;
                pterm.y = pterm.y - (int)pterm.y;            
                pterm.y += 1.f; 
                pterm.y = pterm.y - (int)pterm.y;                  

                part[fj] = pterm;
            }
        } // If this thread is inside the domain in Y
    } // If this thread is inside the domain in X
}
}