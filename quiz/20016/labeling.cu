#include "labeling.h"
#include <cuda.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define MAXN 10000000
#define K 500
#define BLOCKSIZE 1000
#define USE_THRUST

__global__ void cudaLabeling(const char *cuStr, int *cuPos, int strLen){
    // [startidx, endidx) block
    // startidx: true idx of this block, endidx: true end idx of this block
    int startidx = blockIdx.x * BLOCKSIZE;
    int endidx = ((startidx + BLOCKSIZE) <= strLen)? (startidx + BLOCKSIZE):(strLen);
    int in_range_ofEndIdx = (startidx + threadIdx.x < endidx);
    __shared__ int sharedPos[2][BLOCKSIZE]; // double buffering
    __shared__ int p_startidx;
    // search left side of this block TODO: This part will bound performance(workaround: use block size 1024 to resolve this)
    if(threadIdx.x == 0){
        p_startidx = 0;
        for(int i = 1; i < K; i++){
            // short-circuit logic
            if(startidx - i >= 0 && cuStr[startidx - i] != ' '){
                p_startidx++;
            }else break;
        }
    }
    if(in_range_ofEndIdx){
        sharedPos[0][threadIdx.x] = (cuStr[startidx + threadIdx.x] != ' '); // whether this position is a nonspace char
    } else sharedPos[0][threadIdx.x] = 0;
    __syncthreads(); // make sure all the thread in this block sync
    int oldBufIdx = 0, newBufIdx = 1, temp; // [*] Double buffering technique
    // Tree
    for(int stride = 1; stride <= 256; stride *= 2){
        if((int)(threadIdx.x) - stride >= 0 && sharedPos[oldBufIdx][threadIdx.x] == stride){
            // Only if the continuos character is equal to stride, we will add the value into it
            sharedPos[newBufIdx][threadIdx.x] = sharedPos[oldBufIdx][threadIdx.x] + sharedPos[oldBufIdx][threadIdx.x - stride];
        }else sharedPos[newBufIdx][threadIdx.x] = sharedPos[oldBufIdx][threadIdx.x];
        __syncthreads();
        temp = oldBufIdx;
        oldBufIdx = newBufIdx;
        newBufIdx = temp;
    }
    // Add all thread with `p_startidx`
    if(in_range_ofEndIdx){
        if(sharedPos[oldBufIdx][threadIdx.x] > 0 && (threadIdx.x + 1) == sharedPos[oldBufIdx][threadIdx.x]){
            cuPos[startidx + threadIdx.x] = p_startidx + sharedPos[oldBufIdx][threadIdx.x];
        }else{
            cuPos[startidx + threadIdx.x] = sharedPos[oldBufIdx][threadIdx.x];
        }
    }
}

__global__ void cudaLabeling_simple(const char *cuStr, int *cuPos, int strLen){
    // [startidx, endidx) block
    // startidx: true idx of this block, endidx: true end idx of this block
    int startidx = blockIdx.x * BLOCKSIZE;
    int endidx = ((startidx + BLOCKSIZE) <= strLen)? (startidx + BLOCKSIZE):(strLen);
    // search left side of this block
    int p_startidx = 0;
    for(int i = 0; i < K; i++){
        // short-circuit logic
        if(startidx - i >= 0 && cuStr[startidx - i] != ' '){
            p_startidx++;
        }else break;
    }
    cuPos[startidx] = p_startidx;
    int prev = p_startidx;
    for(int i = startidx+1; i < endidx; i++){
        if(cuStr[i] == ' '){
            prev = 0;
            cuPos[i] = 0;
        }
        else{
            cuPos[i] = prev + 1;
            prev = prev + 1;
        }
    }
}
template<class T> struct MM{
    const char *cuStr;
    MM(const char *cuStr_): cuStr(cuStr_) {};
    __host__ __device__ T operator()(const T &index) const{
        if(cuStr[index] != ' ') return -1;
        return index;
    };
};
// TODO: May be dangerous
template<class T> struct IndexSubtractValueOp{
    int *p;
    IndexSubtractValueOp(int *p_): p(p_){};
    __host__ __device__ T operator()(const T &index) const{
        return index - p[index];
    };
};
void labeling(const char *cuStr, int *cuPos, int strLen){
    int blockSize = BLOCKSIZE; // a thread will compute `blockSize` elements
    int blocksPerGrid = (strLen + blockSize - 1) / blockSize; // how many blocks do we have ( ceil() function)
    int threadsPerBlock = 1; // each block will be only computed by one thread

#ifdef USE_THRUST
    static thrust::device_vector<int> p(MAXN);
    thrust::device_ptr<int> dev_ptr_pos(cuPos);
    // subtract index
    thrust::tabulate(thrust::device, p.begin(), p.begin()+strLen, MM<int>(cuStr));
    // prefix maximum
    thrust::inclusive_scan(thrust::device, p.begin(), p.begin() + strLen, p.begin(), thrust::maximum<int>());
    int *raw_p_ptr = thrust::raw_pointer_cast(p.data()); // TODO: May be dangerous
    // index subtract value
    thrust::tabulate(thrust::device, p.begin(), p.begin()+strLen, IndexSubtractValueOp<int>(raw_p_ptr));
    thrust::copy(thrust::device, p.begin(), p.begin() + strLen, dev_ptr_pos);
#else
    cudaLabeling_simple<<< blocksPerGrid, threadsPerBlock >>>(cuStr, cuPos, strLen);
#endif



}
