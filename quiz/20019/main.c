#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#include <string.h>
#include <inttypes.h>
#include <omp.h>
#define MAXBUF 1000
#define MAXCPU 100
#define MAXK 5000
#define MAXLOG 5000
#define MAXN_POWER 36
#define CHUNK_POWER 10  // 8 will not pass the time limit!!!!
// chunk_size = 2^chunk_power
#define CHUNK_SIZE (1 << (CHUNK_POWER)) 
// maxgroup = 2^(maxn_power - chunk_power)
#define MAXGROUP (1 << (MAXN_POWER-CHUNK_POWER))
#define GROUP_STRIDE 256
#define DEVICE_NUM 2
cl_int status;

typedef struct Keys_{
    cl_uint key1[1];
    cl_uint key2[1];
} Keys;
void platform(cl_platform_id *platform_id, cl_uint *platform_id_got){
    status = clGetPlatformIDs(1, platform_id, 
                    platform_id_got);
    assert(status == CL_SUCCESS && *platform_id_got == 1);
#ifdef DEBUG
    printf("%d platform found\n", *platform_id_got);
#endif
}
void getDevice(cl_platform_id *platform_id, cl_device_id CPU[], cl_uint *CPU_id_got){
#ifndef NOTUSEGPU
    status = clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_GPU, 
                  MAXCPU, CPU, CPU_id_got);
#else
    status = clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_CPU, 
                  MAXCPU, CPU, CPU_id_got);
#endif
    assert(status == CL_SUCCESS);
#ifdef DEBUG
    printf("There are %d CPU devices\n", *CPU_id_got); 
#endif
}
void getContext(cl_uint *CPU_id_got, cl_device_id CPU[], cl_context *context){
    *context = clCreateContext(NULL, DEVICE_NUM, CPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
#ifdef DEBUG
    printf("Context Success\n");
#endif
}
void createCommandQueue(cl_context *context, cl_device_id *device_id, cl_command_queue *commandQueue){
    *commandQueue = clCreateCommandQueue(*context, *device_id, 0, &status);
    assert(status == CL_SUCCESS);
#ifdef DEBUG
    printf("Command Queue Success");
#endif
}
void load_my_code(char *filename, cl_program *program, cl_context *context){
    FILE *kernelfp = fopen(filename, "r");
    assert(kernelfp != NULL);

    char kernelBuffer[MAXK];
    const char *constKernelSource = kernelBuffer;
    size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
    *program = 
        clCreateProgramWithSource(*context, 1, &constKernelSource, 
			      &kernelLength, &status);
    assert(status == CL_SUCCESS);
#ifdef DEBUG
    printf("Create Program Success\n");
#endif
}
void buildProgram(cl_program *program, cl_uint *CPU_id_got, cl_device_id CPU[]){
    assert(DEVICE_NUM <= *CPU_id_got);
    status = 
        clBuildProgram(*program, DEVICE_NUM, CPU, NULL, NULL, 
               NULL);
}
void getbuildProgramInfo(cl_program *program, cl_device_id *device_id, char logInformation[], size_t logInfoByteSize){
    size_t return_length;
    status = clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, logInfoByteSize, logInformation, &return_length);
    // Append null character 
    logInformation[return_length/sizeof(logInformation[0])] = '\0';
}
void buildKernel(cl_program *program, const char *function_name, cl_kernel *kernel){
    *kernel = clCreateKernel(*program, function_name, &status);
    assert(status == CL_SUCCESS);
}

void buildBuffers(cl_context *context, cl_mem *arrayBuffer, cl_uint array[]){
    // TODO: Maybe keys->key1 is not a pointer?
    *arrayBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, MAXGROUP * sizeof(array[0]), array, &status);
    assert(status == CL_SUCCESS);
}
void static_link_params(cl_kernel *kernel, cl_mem *arrayBuffer){
    status = clSetKernelArg(*kernel, 2, sizeof(cl_uint) * CHUNK_SIZE, NULL); // local memory
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(*kernel, 3, sizeof(cl_mem), (void *)arrayBuffer);
    assert(status == CL_SUCCESS);
}
void dynamic_link_params(cl_kernel *kernel, Keys *keys, cl_int *N){
    status = clSetKernelArg(*kernel, 0, sizeof(keys->key1[0]), (void *)&keys->key1[0]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(*kernel, 1, sizeof(keys->key2[0]), (void *)&keys->key2[0]);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(*kernel, 4, sizeof(N[0]), (void *)&N[0]);
    assert(status == CL_SUCCESS);
}
void enqueueCommand(cl_command_queue *commandQueue, cl_kernel *kernel, int dim, size_t *global_work_size, size_t *local_work_size, size_t *group_number, cl_int N[]){
    // save group numbers
    for(int i = 0; i < dim; i++){
        // ceil
        if(N[i] % local_work_size[i] != 0){
            group_number[i] = (N[i] / local_work_size[i]) + 1;
        }
        else group_number[i] = N[i] / local_work_size[i];
        
        global_work_size[i] = group_number[i] * local_work_size[i];
#ifdef DEBUG
        printf("%lu\n", group_number[i]);
#endif
    }

    // group padding 
    if(group_number[0] % GROUP_STRIDE != 0){
        int num_pad_group = GROUP_STRIDE - (group_number[0] % GROUP_STRIDE);
        global_work_size[0] = local_work_size[0] * (group_number[0] + num_pad_group) / GROUP_STRIDE;
        group_number[0] += num_pad_group; // add the number of padding group
    }else{
        global_work_size[0] = local_work_size[0] * group_number[0] / GROUP_STRIDE;
    }
    status = clEnqueueNDRangeKernel(*commandQueue, *kernel, dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
}
void readResult(cl_command_queue *commandQueue, cl_mem *arrayBuffer, cl_uint array[]){
    status = clEnqueueReadBuffer(*commandQueue, *arrayBuffer, CL_TRUE, 0, MAXGROUP * sizeof(array[0]), array, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
}
void updateBuffer(cl_command_queue *commandQueue, cl_mem *keybuffer1, cl_mem *keybuffer2, cl_mem *NBuffer, 
                Keys *keys, cl_int N[]){
    status = clEnqueueWriteBuffer(*commandQueue, *keybuffer1, CL_TRUE, 0, 1 * sizeof(keys->key1[0]), keys->key1, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueWriteBuffer(*commandQueue, *keybuffer2, CL_TRUE, 0, 1 * sizeof(keys->key2[0]), keys->key2, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
    status = clEnqueueWriteBuffer(*commandQueue, *NBuffer, CL_TRUE, 0, 1 * sizeof(N[0]), N, 0, NULL, NULL);
    assert(status == CL_SUCCESS);
}
void releaseAll(cl_uint *array, cl_context *context, cl_command_queue *commandQueue, 
        cl_program *program, cl_kernel *kernel, cl_mem *arrayBuffer){
    free(array);
    clReleaseContext(*context);
    clReleaseCommandQueue(*commandQueue);
    clReleaseProgram(*program);
    clReleaseKernel(*kernel);
    clReleaseMemObject(*arrayBuffer);
}
int main(){
    char filename[MAXBUF] = "vecdot.cl";
    char logInformation[MAXLOG];
    cl_platform_id platform_id;
    cl_uint platform_id_got;
    cl_device_id CPU[MAXCPU];
    cl_uint CPU_id_got;
    cl_context context;
    cl_command_queue commandQueue[DEVICE_NUM];
    cl_program program;
    cl_kernel kernel[DEVICE_NUM];  
    cl_mem arrayBuffer;
    size_t global_work_size[1];
    size_t local_work_size[1];
    size_t group_number[1];
#ifdef DEBUG
    cl_uint *array = (cl_uint*) calloc(MAXGROUP, sizeof(cl_uint));
#else
    cl_uint *array = (cl_uint*) malloc(sizeof(cl_uint) * MAXGROUP);
#endif
    Keys keys;
    cl_int N[1];
    size_t len = strlen(filename);
    assert(len > 0);
    if(filename[len-1] == '\n') filename[len-1] = '\0';
#ifdef DEBUG
    printf("filename: %s\n", filename);
#endif
    // [*] Get platform related information
    platform(&platform_id, &platform_id_got);
    // [*] Get device related information
    getDevice(&platform_id, CPU, &CPU_id_got);
    // [*] Get context
    getContext(&CPU_id_got, CPU, &context);
    // [*] Create commandQueue
    for(int i = 0; i < DEVICE_NUM; i++){
        createCommandQueue(&context, &CPU[i], &commandQueue[i]);
    }
    // [*] Load code
    load_my_code(filename, &program, &context);
    // [*] Build code
    buildProgram(&program, &CPU_id_got, CPU);
#ifdef DEBUG
    getbuildProgramInfo(&program, &CPU[0], logInformation, sizeof(logInformation));
    printf("%s\n", logInformation);
#endif 
    // [*] Create kernel
    for(int i = 0; i < DEVICE_NUM; i++){
        buildKernel(&program, "vecdot_seq_reduce_half", &kernel[i]);
    }
    // [*] Create buffer
    // TODO: DEBUG
    buildBuffers(&context, &arrayBuffer, array);
    // [*] Parameter linking
    for(int i = 0; i < DEVICE_NUM; i++){
        static_link_params(&kernel[i], &arrayBuffer);
    }
    // 
    while(scanf("%d %" PRIu32 " %" PRIu32, &N[0], &keys.key1[0], &keys.key2[0]) == 3){
#ifdef DEBUG
        printf("%d %u %u\n", N[0], keys.key1[0], keys.key2[0]);
#endif
        local_work_size[0] = CHUNK_SIZE;

        // [*] WARNING: remove this line will cause bug!(because of caching on devices)
        dynamic_link_params(&kernel[0], &keys, N);
        // [*] Enqueue command
        enqueueCommand(&commandQueue[0], &kernel[0], 1, global_work_size, local_work_size, group_number, N);
        // [*] Read 
        readResult(&commandQueue[0], &arrayBuffer, array);
        // Sum over array
        int group_num = group_number[0];
        uint32_t sum = 0;
#pragma omp parallel for reduction(+: sum)
        for(int i = 0; i < group_num; i += GROUP_STRIDE){
            sum += array[i];
#ifdef DEBUG
            printf("array[%d]: %u\n", i, array[i]);
#endif
        }
        // [*] Check if there is a padding group
        printf("%" PRIu32 "\n", sum);
    }
    releaseAll(array, &context, &commandQueue[0], &program, &kernel[0], &arrayBuffer);
    return 0;
}
