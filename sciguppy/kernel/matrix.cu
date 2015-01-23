// TODO: Make more generic
__global__ void subset_assignment_kernel(float *d_a, float *d_b, int a_x, int a_width, int size) {
    // Get the id and make sure it is within bounds
    const int b_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (b_id >= size) {
        return;
    }

    const int a_id = a_x * size + b_id;
    d_a[a_id] = d_b[b_id];
}
