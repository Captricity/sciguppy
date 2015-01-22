__global__ void elementwise_div_kernel(float *d_a, float s, float *d_out, int size) {
    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    d_out[id] = d_a[id] / s;
}
