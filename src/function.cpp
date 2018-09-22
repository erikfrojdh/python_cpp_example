#include <iostream>

double* add(const double *a, const double *b, double *result, int size){

    //allocate output
    for (int i=0; i!=size; ++i){
        result[i] = a[i] + b[i];
    }
    return result;

}