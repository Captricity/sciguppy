__global__ void expit_kernel(float *d_a, float *d_aout, int size) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = d_a[id];
    float tmp;
    if (x < 0) {
        tmp = expf(x);
        d_aout[id] = tmp / (1.0 + tmp);
    } else {
        d_aout[id] = 1.0 / (1.0 + expf(-d_a[id]));
    }
}

__global__ void expit_fast_kernel(float *d_a, float *d_aout, int size) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = d_a[id];
    float tmp;
    if (x <= -6) {
        d_aout[id] = 0;
    } else if (x >= 6) {
        d_aout[id] = 1;
    } else if (x < 0) {
        tmp = __expf(x);
        d_aout[id] = tmp / (1.0 + tmp);
    } else {
        d_aout[id] = 1.0 / (1.0 + __expf(-d_a[id]));
    }
}

__global__ void expit_back_kernel(float *d_a, float *d_err, float *d_out, int size) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = d_a[id];
    d_out[id] = x * (1-x) * d_err[id];
}
