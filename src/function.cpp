void cpp_add(const double *a, const double *b, double *result, int size){
    for (int i=0; i!=size; ++i){
        result[i] = a[i] + b[i];
    }
}