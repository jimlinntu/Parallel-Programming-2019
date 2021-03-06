#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#define UINT uint32_t
#define MAXN 1024
#define MAXGPU 2
#define MAX_TESTCASE 512
#define THREADS_PER_BLOCK 16
#define USE_TRANSPOSE
// in-place transpose
__global__ void cudaTranspose(int N, UINT target[][MAXN]){
    // Share works among row index
    // row index of target matrix
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    UINT temp;
    // swap until diagonal index reaches
    for(int col = 0; col < index_x; col++){
        temp = target[index_x][col];
        target[index_x][col] = target[col][index_x];
        target[col][index_x] = temp;
    }
}
// multiple src1 and src2
// TODO: Fix out of range multiplication error
__global__ void cudaMultiply(int N, UINT src1[][MAXN], UINT src2[][MAXN], UINT target[][MAXN]){
    UINT sum = 0;
    int x = blockIdx.x * blockDim.x + threadIdx.x; // target's x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // target's y
    __shared__ UINT local_src1[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ UINT local_src2[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    // loop over each block
    for(int i = 0; i < gridDim.x; i++){
        if(x < N && i * blockDim.y + threadIdx.y < N){
            local_src1[threadIdx.x][threadIdx.y] = src1[x][i * THREADS_PER_BLOCK + threadIdx.y];
        }else{
            // padding
            local_src1[threadIdx.x][threadIdx.y] = 0;
        }
        if(i * blockDim.x + threadIdx.x < N && y < N){
#ifndef USE_TRANSPOSE
            local_src2[threadIdx.x][threadIdx.y] = src2[i * THREADS_PER_BLOCK + threadIdx.x][y];
#else
            // Transpose version
            local_src2[threadIdx.y][threadIdx.x] = src2[i * THREADS_PER_BLOCK + threadIdx.x][y];
#endif

        }else{
            //padding
#ifndef USE_TRANSPOSE
            local_src2[threadIdx.x][threadIdx.y] = 0;
#else
            // Transpose padding
            local_src2[threadIdx.y][threadIdx.x] = 0;
#endif
        }
        __syncthreads();
        // local matrix multiplication
        for(int j = 0; j < THREADS_PER_BLOCK; j++){
#ifndef USE_TRANSPOSE
            sum += local_src1[threadIdx.x][j] * local_src2[j][threadIdx.y];
#else
            sum += local_src1[threadIdx.x][j] * local_src2[threadIdx.y][j];
#endif
        }
        // Move to next block 
        __syncthreads();
    }
    target[x][y] = sum;
}
__global__ void cudaAdd(int N, UINT src1[][MAXN], UINT src2[][MAXN], UINT target[][MAXN]){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    target[x][y] = src1[x][y] + src2[x][y];
}

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}
void print_matrix(int N, UINT A[][MAXN]) {
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[");
        for (int j = 0; j < N; j++)
            fprintf(stderr, " %u", A[i][j]);
        fprintf(stderr, " ]\n");
    }
}
UINT signature(int N, UINT A[][MAXN]) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}
void device2host(UINT cudaMem[][MAXN], UINT hostMem[][MAXN]){
    cudaError_t error;
    error = cudaMemcpy(hostMem, cudaMem, sizeof(UINT)*MAXN*MAXN, cudaMemcpyDeviceToHost);
    if(error == cudaErrorInvalidValue){
        fprintf(stderr, "cudaErrorInvalidValue\n");
    }else if(error == cudaErrorInvalidDevicePointer){
        fprintf(stderr, "cudaErrorInvalidDevicePointer\n");
    }else if(error == cudaErrorInvalidMemcpyDirection){
        fprintf(stderr, "cudaErrorInvalidMemcpyDirection\n");
    }
    assert(error == cudaSuccess);
}
void host2device(UINT cudaMem[][MAXN], UINT hostMem[][MAXN]){
    cudaError_t error;
    error = cudaMemcpy(cudaMem[0], hostMem[0], sizeof(UINT)*MAXN*MAXN, cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
}

typedef struct Signature_{
    UINT X,Y;
} Signature;

Signature signatures[MAX_TESTCASE];
UINT X[MAXGPU][MAXN][MAXN], Y[MAXGPU][MAXN][MAXN];
UINT A[MAXGPU][MAXN][MAXN], B[MAXGPU][MAXN][MAXN];
UINT (*cudaX[MAXGPU])[MAXN], (*cudaY[MAXGPU])[MAXN];
UINT (*cudaA[MAXGPU])[MAXN], (*cudaB[MAXGPU])[MAXN];
UINT (*cudaAB[MAXGPU])[MAXN], (*cudaBA[MAXGPU])[MAXN];
UINT (*cudaABA[MAXGPU])[MAXN], (*cudaBAB[MAXGPU])[MAXN];
int main() {
    int N[MAX_TESTCASE], S[MAX_TESTCASE][2]; // TODO: May have bug because S_i <= 2^31
    int testcase_num = 0;
    omp_set_num_threads(MAXGPU);
    cudaError_t ret;
#ifdef DEBUG
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("DeviceCount: %d\n", count);
    struct cudaDeviceProp prop;
    for(int i = 0; i < MAXGPU; i++){
        printf("Device %d\n:", i);
        cudaGetDeviceProperties(&prop, i);
        printf("maxThreadsDim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
        assert(THREADS_PER_BLOCK * THREADS_PER_BLOCK <= prop.maxThreadsPerBlock);
    }
#endif
    // allocate memory to each GPU device
    for(int i = 0; i < MAXGPU; i++){
        ret = cudaSetDevice(i);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaX[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaY[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaA[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaB[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaAB[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaBA[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaABA[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
        ret = cudaMalloc(&cudaBAB[i], sizeof(UINT) * MAXN * MAXN);
        assert(ret == cudaSuccess);
    }
    for(testcase_num = 0; scanf("%d", &N[testcase_num]) == 1; testcase_num++){
        scanf("%d", &S[testcase_num][0]);
        scanf("%d", &S[testcase_num][1]);
    }
    assert(testcase_num <= MAX_TESTCASE);
    // Multi-GPU load balancing
#pragma omp parallel for schedule(dynamic, 1) 
    for(int i = 0; i < testcase_num; i++){
        int currentGPU = omp_get_thread_num();
        assert(currentGPU < MAXGPU);
#ifdef DEBUG
        printf("currentGPU: %d\n", currentGPU);
#endif
        ret = cudaSetDevice(currentGPU);
        assert(ret == cudaSuccess);
        int n = N[i], s_a = S[i][0], s_b = S[i][1];
        rand_gen(s_a, n, A[currentGPU]);
        rand_gen(s_b, n, B[currentGPU]);
        host2device(cudaA[currentGPU], A[currentGPU]);
        host2device(cudaB[currentGPU], B[currentGPU]);
#ifdef DEBUG
        memset(A[currentGPU], 0, sizeof(UINT) * MAXN * MAXN);
        memset(B[currentGPU], 0, sizeof(UINT) * MAXN * MAXN);
        device2host(cudaA[currentGPU], A[currentGPU]);
        device2host(cudaB[currentGPU], B[currentGPU]);
        printf("Matrix A:\n");
        print_matrix(n, A[currentGPU]);
        printf("Matrix B:\n");
        print_matrix(n, B[currentGPU]);
#endif
        int blocksPerGrid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int threadsPerBlock = THREADS_PER_BLOCK;
        dim3 dimBlock(threadsPerBlock, threadsPerBlock);
        dim3 dimGrid(blocksPerGrid, blocksPerGrid);
        // AB
        cudaMultiply<<< dimGrid, dimBlock >>>(n, cudaA[currentGPU], cudaB[currentGPU], cudaAB[currentGPU]);
        // BA
        cudaMultiply<<< dimGrid, dimBlock >>>(n, cudaB[currentGPU], cudaA[currentGPU], cudaBA[currentGPU]);
        // AB+BA
        // Note: default stream will sync, so this is redundant
        cudaDeviceSynchronize(); // make sure AB, BA are done
        cudaAdd<<< dimGrid, dimBlock >>>(n, cudaAB[currentGPU], cudaBA[currentGPU], cudaX[currentGPU]);
        // ABA
        cudaMultiply<<< dimGrid, dimBlock >>>(n, cudaAB[currentGPU], cudaA[currentGPU], cudaABA[currentGPU]);
        // BAB
        cudaMultiply<<< dimGrid, dimBlock >>>(n, cudaBA[currentGPU], cudaB[currentGPU], cudaBAB[currentGPU]);
        // ABA+BAB
        cudaDeviceSynchronize();
        cudaAdd<<< dimGrid, dimBlock >>>(n, cudaABA[currentGPU], cudaBAB[currentGPU], cudaY[currentGPU]);
        // [*] Print results
        device2host(cudaX[currentGPU], X[currentGPU]);
        device2host(cudaY[currentGPU], Y[currentGPU]);
        signatures[i].X = signature(n, X[currentGPU]);
        signatures[i].Y = signature(n, Y[currentGPU]);
    }
    for(int i = 0; i < testcase_num; i++){
        printf("%u\n", signatures[i].X);
        printf("%u\n", signatures[i].Y);
    }
    for(int i = 0; i < MAXGPU; i++){
        cudaFree(cudaX[i]);
        cudaFree(cudaY[i]);
        cudaFree(cudaA[i]);
        cudaFree(cudaB[i]);
        cudaFree(cudaAB[i]);
        cudaFree(cudaBA[i]);
        cudaFree(cudaABA[i]);
        cudaFree(cudaBAB[i]);
    }
    return 0;
}
