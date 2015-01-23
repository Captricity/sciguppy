__global__ void rectify_back_kernel(float *d_a, float *d_error, float *d_out, int size) {
    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    const float x = d_a[id];
    if (x > 0) {
        d_out[id] = d_error[id];
    } else {
        d_out[id] = 0;
    }
}
