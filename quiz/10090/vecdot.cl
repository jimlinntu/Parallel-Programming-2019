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
__kernel void vecdot_seq(__global uint *key1, __global uint *key2, __local uint *localarray, __global uint *output, __global int *N){
    int idx = get_global_id(0);
    int group_id = get_group_id(0);
    int local_idx = get_local_id(0);
    int local_size = get_local_size(0);
    int tempN = N[0];
    // Initialize
    if(idx < tempN) localarray[local_idx] = encrypt((uint)idx, key1[0]) * encrypt((uint)idx, key2[0]);
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
__kernel void vecdot_seq_reduce_half(__global uint *key1, __global uint *key2, __local uint *localarray, __global uint *output, __global int *N){
    int idx = get_global_id(0);
    int group_id = get_group_id(0);
    int local_idx = get_local_id(0);
    int local_size = get_local_size(0);
    int tempN = N[0];
    // Initialize
    if(idx < tempN){
        localarray[local_idx] = encrypt((uint)idx, key1[0]) * encrypt((uint)idx, key2[0]);
        if(idx + local_size < tempN){
            localarray[local_idx] += encrypt((uint)(idx+local_size), key1[0]) * encrypt((uint)(idx+local_size), key2[0]);
        }
    }
    else localarray[local_idx] = 0; // padding
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s = local_size/2; s > 0; s >>= 1){
        if(local_idx < s){
            localarray[local_idx] += localarray[local_idx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // root of local 
    if(local_idx == 0){
        output[group_id] = (group_id % 2 == 0)? localarray[0]:0;
    }
}
