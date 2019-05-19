//typedef unsigned int uint32_t; // be careful! if you set wrong type, the parameter will be wrong when you access pointer!!!!
static inline uint rotate_left(uint x, uint n) {
    return  (x << n) | (x >> (32-n));
}
static inline uint encrypt(uint m, uint key) {
    return (rotate_left(m, key&31) + key)^key;
}
__kernel void vecdot(__global uint *key1, __global uint *key2, __local uint *localarray, __global uint *output, __global int *N){
    int idx = get_global_id(0);
    int group_id = get_group_id(0);
    int local_idx = get_local_id(0);
    int local_size = get_local_size(0);
    int tempN = N[0];
    // Initialize
    if(idx < tempN) localarray[local_idx] = encrypt((uint)idx, key1[0]) * encrypt((uint)idx, key2[0]);
    else localarray[local_idx] = 0; // padding
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = 1; s < local_size; s *= 2){
        if(local_idx % (2 * s)  == 0 && (local_idx + s) < local_size){
            localarray[local_idx] += localarray[local_idx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // root of local 
    if(local_idx == 0) output[group_id] = localarray[0];
}
__kernel void vecdot_seq(__private uint key1, __private uint key2, __local uint *localarray, __global uint *output, __private int N){
    int idx = get_global_id(0);
    int group_id = get_group_id(0);
    int local_idx = get_local_id(0);
    int local_size = get_local_size(0);
    int tempN = N;
    // Initialize
    if(idx < tempN) localarray[local_idx] = encrypt((uint)idx, key1) * encrypt((uint)idx, key2);
    else localarray[local_idx] = 0; // padding
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = local_size/2; s > 0; s >>= 1){
        if(local_idx < s){
            localarray[local_idx] += localarray[local_idx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // root of local 
    if(local_idx == 0) output[group_id] = localarray[0];
}

// [*] This should be same as `main.c`
#define GROUP_STRIDE 256
__kernel void vecdot_seq_reduce_half(__private const uint key1, __private const uint key2, __local uint *localarray, __global uint *output, __private const int N){
    int group_id = get_group_id(0) * GROUP_STRIDE; // group 0, group 2, group 4, ...
    int local_size = get_local_size(0);
    int local_idx = get_local_id(0);
    // NOTE: this idx is NOT equal to global_idx
    int idx = group_id * local_size + local_idx; // ex. (group 0) * local_size + local_idx == real global idx
    int tmpidx = idx - local_size;
    // Initialize
    localarray[local_idx] = 0;
    // [*] This loop will add group that are in the distance of GROUP_STRIDE
    for(int tmp_gidx = 0; tmp_gidx < GROUP_STRIDE; tmp_gidx++){
        tmpidx += local_size;
        if(tmpidx >= N) break;
        localarray[local_idx] += encrypt((uint)tmpidx, key1) * encrypt((uint)tmpidx, key2);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = local_size/2; s > 0; s >>= 1){
        if(local_idx < s){
            localarray[local_idx] += localarray[local_idx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // root of local 
    if(local_idx == 0){
        output[group_id] = localarray[0];
    }
}
