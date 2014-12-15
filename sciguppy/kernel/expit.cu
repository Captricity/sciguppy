__global__ void expit_kernel(float *a, float *aout, int size) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = a[id];
    float tmp;
    if (x < 0) {
        tmp = expf(x);
        aout[id] = tmp / (1.0 + tmp);
    } else {
        aout[id] = 1.0 / (1.0 + expf(-a[id]));
    }
}

__global__ void expit_fast_kernel(float *a, float *aout, int size) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = a[id];
    float tmp;
    if (x <= -6) {
        aout[id] = 0;
    } else if (x >= 6) {
        aout[id] = 1;
    } else if (x < 0) {
        tmp = __expf(x);
        aout[id] = tmp / (1.0 + tmp);
    } else {
        aout[id] = 1.0 / (1.0 + __expf(-a[id]));
    }
}
