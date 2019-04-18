#include <stdio.h>

#define MAXN 10001
#define MAXM 1000001
#define MAX(a,b) ((a)>(b)?(a):(b))
typedef unsigned long long ULL;
typedef struct Thing_{
    int w, v;
} Thing;
int knapsack(int N, int M, Thing thing[]){
    int dptable[MAXM] = {0};
    int dptable_after[MAXM] = {0};
    for(int i = 0; i < N; i++){
        // [*] Avoid excessive indexing
        int thing_w = thing[i].w;
        int thing_v = thing[i].v;
//#pragma omp parallel for
//        for(int remainder = 0; remainder < thing_w; remainder++){
//            for(int w = M - remainder; w >= 0; w -= thing_w){
//                if(thing_w <= w){
//                    dptable[w] = MAX(dptable[w], dptable[w - thing_w] + thing_v);
//                }
//            }
//        }
#pragma omp parallel
        {
#pragma omp for
            for(int w = M; w >= 0; w--){
                if(thing_w > w){
                    dptable_after[w] = dptable[w];
                }else{
                    dptable_after[w] = MAX(dptable[w], dptable[w - thing_w] + thing_v);
                }
            }
#pragma omp for
            for(int w = M; w >= 0; w--){
                dptable[w] = dptable_after[w];
            }
        }
    }
    return dptable[M];
}
int main(){
    int N, M;
    Thing thing[MAXN];
    scanf("%d %d", &N, &M);
    for(int i = 0; i < N; i++){
        scanf("%d %d", &(thing[i].w), &(thing[i].v));
    }
    printf("%d\n", knapsack(N, M, thing));
    return 0;
}
