#include "sort.h"

void swap(float* arr, int i, int j)
{
    float temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
 
float* bubbleSort(float *arr_1[], float *arr_2[], float *arr_3[], int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)
        for (j = 0; j < n - i - 1; j++)
            if ((arr_1[j]) > (arr_1[j + 1]))
                swap(arr_1, j, j + 1);
                swap(arr_2, j, j + 1);
                swap(arr_3, j, j + 1);
              
    float* sort_arry[3];
    sort_arry[0] = arr_1;
    sort_arry[1]  = arr_2;
    sort_arry[2]  = arr_3;
    return  sort_arry;
}