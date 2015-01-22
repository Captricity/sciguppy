__global__ void softmax_back_kernel(float *d_a, float *d_error, float *d_out, float s, int size) {
    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size) {
        return;
    }

    d_out[id] = d_a[id] * (d_error[id] - s);
}
