// TODO: Make more generic
__global__ void subset_assignment_kernel(float *d_a, float *d_b, int a_x, int size) {
    // Get the id and make sure it is within bounds
    const int b_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (b_id >= size) {
        return;
    }

    const int a_id = a_x * size + b_id;
    d_a[a_id] = d_b[b_id];
}

// TODO: Make more generic
__global__ void subset_slice_assignment_kernel(float *d_a, float *d_b, int a_x_start, int size, int non_width) {
    // Get the id and make sure it is within bounds
    const int b_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (b_id >= size) {
        return;
    }

    const int b_x = b_id / non_width;
    const int b_other = b_id % non_width;

    const int a_id = (a_x_start+b_x)*non_width + b_other;
    d_a[a_id] = d_b[b_id];
}

// TODO: Make more generic
__global__ void vector_subset_slice_assignment_kernel(float *d_a, float *d_b, int a_start, int size) {
    // Get the id and make sure it is within bounds
    const int b_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (b_id >= size) {
        return;
    }

    const int a_id = a_start + b_id;
    d_a[a_id] = d_b[b_id];
}
