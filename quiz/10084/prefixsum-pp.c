
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "utils.h"
#include <pthread.h>
#include <assert.h>
#include <math.h>
 
#define MAXN 10000005
#define MAX_THREAD 6
uint32_t prefix_sum[MAXN];
uint32_t key;

typedef struct Params_{
    int start, end, value; 
} Params;

void *thread_prefixsum(void *param_){
    // start point and end point
    Params *param = (Params *)param_;
    int start = param->start, end = param->end;
    uint32_t sum = 0;
    for(int i = start; i < end; i++){
        sum += encrypt(i, key); // collect each element
        prefix_sum[i] = sum; 
    }
    pthread_exit(NULL);
}
void *thread_all_add(void *param_){
    int start = ((Params*)param_)->start;
    int end = ((Params *)param_)->end;
    int value = ((Params *)param_)->value;
    // adding previous last element value
    for(int i = start; i < end; i++){
        prefix_sum[i] += value;
    } 
    pthread_exit(NULL);
}
int main() {
    cpu_set_t cpuset;  
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 6; i++)
        CPU_SET(i, &cpuset);
    assert(sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0);
   
    //
    int n;
    pthread_t threads[MAX_THREAD];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    Params thread_params[MAX_THREAD];
    while (scanf("%d %" PRIu32, &n, &key) == 2) {
        // Divide by MAX_THREAD part
        // note that only last thread will do things if quotient is zero
        int quotient = n / MAX_THREAD;
        for(int i = 0; i < MAX_THREAD; i++){
            // start point and end point
            if(i != MAX_THREAD -1){
                thread_params[i].start = quotient * i + 1;
                thread_params[i].end = quotient * (i + 1) + 1;
            }else{
                thread_params[i].start = quotient * i + 1;
                thread_params[i].end = n + 1;
            }
            int error = pthread_create(&threads[i], &attr, thread_prefixsum, (void *)&thread_params[i]);
        }
        // sync
        for(int i = 0; i < MAX_THREAD; i++){
            pthread_join(threads[i], NULL);
        }
        // thread == number prefix sum
        uint32_t each_thread_prefix[MAX_THREAD];
        // retrieve last element of each block
        for(int i = 0; i < MAX_THREAD; i++){
            if(i != MAX_THREAD-1)
                // last sub prefix
                if(quotient == 0)
                    each_thread_prefix[i] = 0;
                else
                    each_thread_prefix[i] = prefix_sum[quotient * (i + 1)];
            else{
                // last element will is what we want
                each_thread_prefix[i] = prefix_sum[n];
            }
        }
        // naive sequential prefix sum
        uint32_t sum = 0;
        for(int i = 0 ; i < MAX_THREAD; i++){
            sum += each_thread_prefix[i];
            each_thread_prefix[i] = sum; 
        }
        // parallel adding
        for(int i = 1; i < MAX_THREAD; i++){
            thread_params[i].value = each_thread_prefix[i-1];
            if(i != MAX_THREAD -1){
                thread_params[i].start = quotient * i + 1;
                thread_params[i].end = quotient * (i + 1) + 1;
            }else{
                thread_params[i].start = quotient * i + 1;
                thread_params[i].end = n + 1;
            }
            // adding parallel
            int error = pthread_create(&threads[i], &attr, thread_all_add, (void *)&thread_params[i]);
        }
        // Wait untill all added
        for(int i = 1; i < MAX_THREAD; i++){
            pthread_join(threads[i], NULL);
        } 
        output(prefix_sum, n);
    }
    
    pthread_attr_destroy(&attr);
    pthread_exit(NULL);
    return 0;
}
