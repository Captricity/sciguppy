__global__ void expit_kernel(float *a, float *aout, int size, int fast) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = a[id];
    float tmp;
    if (x < 0) {
        if (fast)
            tmp = __expf(x);
        else
            tmp = expf(x);
        aout[id] = tmp / (1.0 + tmp);
    } else {
        if (fast)
            aout[id] = 1.0 / (1.0 + __expf(-a[id]));
        else
            aout[id] = 1.0 / (1.0 + expf(-a[id]));
    }
}
