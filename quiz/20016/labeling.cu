#include "labeling.h"
#include <cuda.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#define MAX(x, y) ((x) < (y))? (y):(x)
#define MAXN 10000000
#define K 500
#define BLOCKSIZE 1024 // 1024 is faster than 512, 2048 cause bug?? (I found that it is because cuda has maximum thread per block)
//#define USE_THRUST


__global__ void cudaLabeling(const char *cuStr, int *cuPos, int strLen){
    // [startidx, endidx) block
    // startidx: true idx of this block, endidx: true end idx of this block
    int startidx = blockIdx.x * BLOCKSIZE;
    int endidx = ((startidx + BLOCKSIZE) <= strLen)? (startidx + BLOCKSIZE):(strLen);
    int in_range_ofEndIdx = (startidx + threadIdx.x < endidx);
    __shared__ int sharedPos[2][BLOCKSIZE]; // double buffering
    __shared__ int p_startidx;
    // search left side of this block (only need to search previous block when this is not an alphabet)
    if(threadIdx.x == 0 && cuStr[startidx] != ' '){
        p_startidx = 0;
        for(int i = 1; i < K; i++){
            // short-circuit logic
            if(startidx - i >= 0 && cuStr[startidx - i] != ' '){
                p_startidx++;
            }else break;
        }
    }
    int prev_shared_pos_cache;
    if(in_range_ofEndIdx){
        prev_shared_pos_cache = sharedPos[0][threadIdx.x] = (cuStr[startidx + threadIdx.x] != ' '); // whether this position is a nonspace char
    }else prev_shared_pos_cache = sharedPos[0][threadIdx.x] = 0;
    __syncthreads(); // make sure all the thread in this block sync
    int oldBufIdx = 0, newBufIdx = 1, temp; // [*] Double buffering technique
    // Tree
    for(int stride = 1; stride <= 256; stride *= 2){
        if((int)(threadIdx.x) - stride >= 0 && prev_shared_pos_cache == stride){
            // Only if the continuos character is equal to stride, we will add the value into it
            prev_shared_pos_cache = sharedPos[newBufIdx][threadIdx.x] = sharedPos[oldBufIdx][threadIdx.x] + sharedPos[oldBufIdx][threadIdx.x - stride];
        }else prev_shared_pos_cache = sharedPos[newBufIdx][threadIdx.x] = sharedPos[oldBufIdx][threadIdx.x];
        __syncthreads();
        temp = oldBufIdx;
        oldBufIdx = newBufIdx;
        newBufIdx = temp;
    }
    // Add all thread with `p_startidx`
    if(in_range_ofEndIdx){
        if((threadIdx.x + 1) == prev_shared_pos_cache){
            cuPos[startidx + threadIdx.x] = p_startidx + prev_shared_pos_cache;
        }else{
            cuPos[startidx + threadIdx.x] = prev_shared_pos_cache;
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
__global__ void cudaLabeling_fast_step_1(const char *cuStr, int *cuPos, int strLen){
    int startidx = blockIdx.x * BLOCKSIZE;
    int endidx = ((startidx + BLOCKSIZE) <= strLen)? (startidx + BLOCKSIZE):(strLen);
    int trueidx = startidx + threadIdx.x;
    __shared__ int R[2][BLOCKSIZE]; // double buffering
    int oldBufIdx = 0, newBufIdx = 1, temp;
    // R0
    R[0][threadIdx.x] = (trueidx < endidx && cuStr[trueidx] == ' ')? (trueidx):(-1);
    __syncthreads();
    // R1
    for(int stride = 1; stride <= 256; stride = stride << 1){
        if((int)threadIdx.x - stride >= 0){
            R[newBufIdx][threadIdx.x] = MAX(R[oldBufIdx][threadIdx.x], R[oldBufIdx][threadIdx.x - stride]);
        }else R[newBufIdx][threadIdx.x] = R[oldBufIdx][threadIdx.x];
        __syncthreads();
        temp = oldBufIdx;
        oldBufIdx = newBufIdx;
        newBufIdx = temp;
    }
    // R2(not fix)
    R[oldBufIdx][threadIdx.x] = (R[oldBufIdx][threadIdx.x] == -1)? (threadIdx.x+1):(trueidx - R[oldBufIdx][threadIdx.x]); // edge case: when value == -1
    if(trueidx < endidx){
        cuPos[trueidx] = R[oldBufIdx][threadIdx.x];
    }
}

__global__ void cudaLabeling_fast_step_2(const char *cuStr, int *cuPos, int strLen){
    int startidx = blockIdx.x * BLOCKSIZE;
    int endidx = ((startidx + BLOCKSIZE) <= strLen)? (startidx + BLOCKSIZE):(strLen);
    if(cuStr[startidx] != ' ' && startidx - 1 >= 0){
        // if two parts of string can meet the boundaries
        if(startidx + threadIdx.x < endidx && threadIdx.x + 1 == cuPos[startidx + threadIdx.x]){
            cuPos[startidx + threadIdx.x] += cuPos[startidx - 1];
        }
    }
    return;
    // [*] Below codes are slower because only one thread will sum up previous block's element
    if(cuStr[startidx] != ' '){
        int prev_sum = 0;
        if(startidx - 1 >= 0){
            // Note: rhs may be 0 !!!!
            prev_sum = cuPos[startidx-1]; // Because two consecutive stream of alphabets will not overlap, we can only add previous block's last element
        }
        if(prev_sum != 0){
            for(int i = startidx; cuStr[i] != ' '; i++){
                cuPos[i] += prev_sum;
            }
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
    int blockSize = BLOCKSIZE; // a thread will compute `blockSize` elements
    int blocksPerGrid = (strLen + blockSize - 1) / blockSize; // how many blocks do we have ( ceil() function)
    int threadsPerBlock = BLOCKSIZE; // each block will be only computed by one thread
//    cudaLabeling<<< blocksPerGrid, threadsPerBlock >>>(cuStr, cuPos, strLen);
//    return;
    cudaLabeling_fast_step_1<<< blocksPerGrid, threadsPerBlock >>>(cuStr, cuPos, strLen);
    // Fix each block
    threadsPerBlock = BLOCKSIZE;
    cudaLabeling_fast_step_2<<< blocksPerGrid, threadsPerBlock >>>(cuStr, cuPos, strLen);
#endif
}
