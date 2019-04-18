#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAXN 19
#define MAXT 3
#define MAX(x,y) ((x)>(y)?(x):(y))
#define ABS(x) ((x)>0?(x):(-(x)))
typedef unsigned long long ULL;
typedef struct Mask_{
    // 0~15 bit represent the positions which are under attacked
    // 0: not attacked
    // 1: attacked
    int row_mask;
    int diag_mask;
    int rdiag_mask;
} Mask; // previous column's mask
// [*] want to put queen at `row` index: return 1 if ok
short ok(const Mask *mask, int row){
    // if row, diagonal, right diagonal are all valid, then return 1
    int pos_mask = (1 << row); // left shift `row` times
    return ((mask->row_mask & pos_mask) == 0 && (mask->diag_mask & pos_mask) == 0 && (mask->rdiag_mask & pos_mask) == 0);
}
// [*] Can be seen as putting queen at `row` index and move forward to next column
Mask update_mask(Mask mask, int row){
    mask.row_mask |= 1 << row; // record this row is occupied
    mask.diag_mask = (mask.diag_mask | (1 << row)) >> 1; // decrease index
    mask.rdiag_mask = (mask.rdiag_mask | (1 << row)) << 1; // increase index
    return mask;
}
// 1: when there is no obstacle, 0: when there is an obstacle
short check_obstacle(int index, ULL obstacle[]){
    return (obstacle[index >> 6] & (1ULL << (index & 63))) == 0ULL;
}
int fastNQueens(int N, int now_col, ULL obstacle[], const Mask *mask){
    // find one solution
    if(now_col >= N) return 1;
    // search
    int sol_num = 0;
    int index;
    Mask local_mask;
    // loop over possible row
    for(int i = 0; i < N; i++){
        index = i * N + now_col;
        // if ok() and no obstacle
        if(ok(mask, i) && check_obstacle(index, obstacle)){
            local_mask = update_mask(*mask, i); // [*] update mask to next row
            sol_num += fastNQueens(N, now_col+1, obstacle, &local_mask);
        }
    }
    return sol_num;
}
int main(){
    char buf[MAXN];
    // Use bit trick to save access obstacle memory's time
    // (from right to left) 0: 0~63, 1: 64~127, 2: 128~191, 3: 192~255
    ULL obstacle[MAXT][4] = {{0}}; 
    // (from right to left) every 4 bits represent `row number`
    int caseNum = 0;
    int N[MAXT];
    while(1){
        fgets(buf, MAXN, stdin);
        if(feof(stdin)) break;
        int localN = N[caseNum] = atoi(buf);
        for(int i = 0; i < localN; i++){
            fgets(buf, MAXN, stdin);
            for(int j = 0; j < localN; j++){
                // obstable[quotient] |= (1 << (remainder))
                int index = localN * i + j;
                if(buf[j] == '*'){
                    obstacle[caseNum][index >> 6] |= (1ULL << (index & 63));
                }
            }
        }
#ifdef DEBUG
        printf("obstacle:\n");
        for(int i = 0; i < 4; i++){
            printf("%llu\n", obstacle[i]);
        }
        printf("===========\n");
#endif
        caseNum++;
    }
    for(int i = 0; i < caseNum; i++){
        int localN = N[i];
        int sol_num = 0;
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:sol_num)
        for(int j = 0; j < localN; j++){
            for(int k = 0; k < localN; k++){
                    int col_0_index = j * localN;
                    int col_1_index = k * localN + 1;
                    // if both obstacle are not valid
                    if(!check_obstacle(col_0_index, obstacle[i]) || 
                            !check_obstacle(col_1_index, obstacle[i])) continue;
                    // if these two queen will attack each other
                    Mask mask = {0, 0 ,0};
                    mask = update_mask(mask, j); // update 0 column mask
                    if(!ok(&mask, k)) continue; // check whether row `k` can be put there
                    mask = update_mask(mask, k); // update 1 column mask
                    sol_num += fastNQueens(localN, 2, obstacle[i], &mask);
            }
        }
        printf("Case %d: %d\n", i+1, sol_num);
    }
    return 0;
}

