#include<stdio.h>
#include<string.h>
#define ULLMAX (0ULL - 1ULL)
#define MAXN 2050
typedef unsigned long long ULL;

int N; // number of matrices
ULL dimensions[MAXN]; // dimensions(i) and dimensions(i+1) correspond to A_i matrix
ULL dptable[MAXN][MAXN]; // dptable(i, j)(or dptable(j, i)) means A_i* ... A_j minimum computing value

ULL matrix_min_chain(ULL dimensions[]){
    // initialize base case: dptable(i, i) and dptable(i, i+1)
    for(int i = 0; i < N; i++){
        // dptable(i, i) = 0
        dptable[i][i] = 0; 
        // dptable(i, i+1) = d_i * d_{i+1} * d_{i+2}
        // dptable(i+1, i) = d_i * d_{i+1} * d_{i+2}
        if(i <= N - 2){
            dptable[i+1][i] = dptable[i][i+1] = dimensions[i] * dimensions[i+1] * dimensions[i+2];
        }
    }
    // DP over subsequence matrix chain's length
#pragma omp parallel // [*] Avoid spawn overhead
    for(int length = 3; length <= N; length++){
        // i: subsequence's starting index
        const int start_point_boundary = N - length; // [*] Avoid conditional statement computing overhead
#pragma omp for
        for(int i = 0; i <= start_point_boundary; i++){
            ULL minimum = ULLMAX;
            ULL temp_sum;
            const int split_point_boundary = i + length - 2; // [*] Avoid conditional statement computing overhead
            const int end_boundary = i + length - 1;
            // [*] Loop over each split point
            // j: split point: i.e. (A_i* ... * A_j) * (A_j+1 * ... * A_{i+length-1})
            for(int j = i; j <= split_point_boundary; j++){
                // [*] In original code, long addressing will cause frequent cache miss 
                // because j index will let dptable(j+1, i+length-1) loop over row, which hurt C row-major memory layout
#ifdef CACHEMISS
                temp_sum = dptable[i][j] + dimensions[i] * dimensions[j+1] * dimensions[i+length] + dptable[j+1][end_boundary];
#else
                temp_sum = dptable[i][j] + dimensions[i] * dimensions[j+1] * dimensions[i+length] + dptable[end_boundary][j+1];
#endif
                if(temp_sum < minimum){
                    minimum = temp_sum;
                }
            }
            // save dp result to dptable(i, i + length - 1)
            dptable[end_boundary][i] = dptable[i][end_boundary] = minimum;
        }
    }

    // return dptable(0, N-1)
    return dptable[0][N-1];
}
int main(){
    while(scanf("%d", &N) != EOF){
        for(int i = 0; i < N+1; i++){
            scanf("%llu", &dimensions[i]);
        }
        printf("%llu\n", matrix_min_chain(dimensions));
    }
}
