#include "MergeSort.hpp"

#include <stdio.h>
#include <stdlib.h>

// Merge two subarrays L and M into arr
void merge(volatile double arr[], uint p, uint q, uint r) {
    // Create L ← A[p..q] and M ← A[q+1..r]
    uint n1 = q - p + 1;
    uint n2 = r - q;

    double L[n1], M[n2];

    for (uint i = 0; i < n1; i++)
        L[i] = arr[p + i];
    for (uint j = 0; j < n2; j++)
        M[j] = arr[q + 1 + j];

    // Maintain current index of sub-arrays and main array
    uint i, j, k;
    i = 0;
    j = 0;
    k = p;

    // Until we reach either end of either L or M, pick larger among
    // elements L and M and place them in the correct position at A[p..r]
    while (i < n1 && j < n2) {
        if (L[i] <= M[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = M[j];
            j++;
        }
        k++;
    }

    // When we run out of elements in either L or M,
    // pick up the remaining elements and put in A[p..r]
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = M[j];
        j++;
        k++;
    }
}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(volatile double arr[], uint l, uint r) {
    if (l < r) {
        // m is the point where the array is divided into two subarrays
        uint m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted subarrays
        merge(arr, l, m, r);
    }
}