#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#include <string.h>
#define MAXBUF 1000
#define MAXCPU 100
#define MAXK 5000
#define MAXLOG 5000
cl_int status;

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
    *context = clCreateContext(NULL, *CPU_id_got, CPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
#ifdef DEBUG
    printf("Context Success\n");
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
    status = 
        clBuildProgram(*program, *CPU_id_got, CPU, NULL, NULL, 
               NULL);
}
void getbuildProgramInfo(cl_program *program, cl_device_id *device_id, char logInformation[], size_t logInfoByteSize){
    size_t return_length;
    status = clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, logInfoByteSize, logInformation, &return_length);
    // Append null character 
    logInformation[return_length/sizeof(logInformation[0])] = '\0';
}
int main(){
    char filename[MAXBUF];
    char logInformation[MAXLOG];
    cl_platform_id platform_id;
    cl_uint platform_id_got;
    cl_device_id CPU[MAXCPU];
    cl_uint CPU_id_got;
    cl_context context;
    cl_program program;
    fgets(filename, MAXBUF, stdin);
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
    // [*] Load code
    load_my_code(filename, &program, &context);
    // [*] Build code
    buildProgram(&program, &CPU_id_got, CPU);
    getbuildProgramInfo(&program, &CPU[0], logInformation, sizeof(logInformation));
    printf("%s", logInformation);
}
