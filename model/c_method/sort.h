#ifndef _SORT_H_
#define _SORT_H_

#ifdef  __cplusplus
extern "C" {
#endif
#include "windows.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

void swap(float* arr, int i, int j);
float* bubbleSort(float* arr_1[], float* arr_2[], float* arr_3[], int n);
typedef struct Sort Sort;

struct Sort
{
    float arr_1;
    float arr_2;
    float arr_3;
};


#ifdef  __cplusplus
}

#endif

#endif  /* _SORT_H_ */