#include <stdio.h>
#include <assert.h>

#define MAXHW 501
#define MIN(x,y) (((a)<(b))?(a):(b))

typedef struct Cord_{
    int h, w;
} Cord;
typedef unsigned long long ULL;
void printMatrix(int matrix[][MAXHW], int h, int w){
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}
Cord findMinDiff(int A[][MAXHW], int B[][MAXHW], int ah, int aw, int bh, int bw){
    int endH = ah - bh; 
    int endW = aw - bw;
    int endIndex = endH * aw + endW;
//#ifdef DEBUG
//    printf("findMinDiff\n");
//    printf("endH: %d, endW: %d\n", endH, endW);
//#endif
    ULL mindiff = (0ULL-1ULL);
    int minIndex = -1;
#pragma omp parallel for
    for(int i = 0; i <= endIndex; i++){
        int x, y;
        ULL diff = 0;
        x = i / aw;
        y = i % aw;
        // [*] Dirty workaround
        if(y > endW){
            continue;
        }
        for(int j = 0; j < bh; j++){
            for(int k = 0; k < bw; k++){
                // (a - b)^2
                diff += (ULL)((A[x+j][y+k] - B[j][k]) * (A[x+j][y+k] - B[j][k]));
            }
        }
//#ifdef DEBUG
//        printf("x: %d, y: %d, diff: %d\n", x, y, diff);
//#endif
        // [*] Only one thread can enter this update
#pragma omp critical
        {
            // if diff < mindiff or mindiff has not yet set
            if(diff < mindiff || mindiff == (0ULL - 1ULL)){
                minIndex = i;
                mindiff = diff;
            }else if(diff == mindiff){
                // choose least index
                if(x < minIndex / aw){
                    minIndex = i;
                    mindiff = diff;
                }else if(x == minIndex / aw){
                    if(y < minIndex % aw){
                        minIndex = i;
                        mindiff = diff;
                    }
                }
            }
        }
    }
    return (Cord){minIndex / aw + 1, minIndex % aw + 1};
}
int main(){
    int ah, aw, bh, bw;
    int A[MAXHW][MAXHW], B[MAXHW][MAXHW];
    while(scanf("%d %d %d %d", &ah, &aw, &bh, &bw) == 4){
        for(int i = 0; i < ah; i++){
            for(int j = 0; j < aw; j++){
               scanf("%d", &A[i][j]);
            }
        }
        for(int i = 0; i < bh; i++){
            for(int j = 0; j < bw; j++){
                scanf("%d", &B[i][j]);
            }
        }
//#ifdef DEBUG
//        printf("PrintMatrix\n");
//        printMatrix(A, ah, aw);
//        printMatrix(B, bh, bw);
//        printf("===========\n");
//#endif
        Cord ansCord = findMinDiff(A, B, ah, aw, bh, bw);
        printf("%d %d\n", ansCord.h, ansCord.w);
    }
    return 0;
}
