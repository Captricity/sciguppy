#include <stdio.h>

__global__ void ewsum_kernel(float *d_a, float *d_w, float *d_out, int num_w, int width, int total_dim) {

    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= total_dim) {
        return;
    }

    const int non_width = total_dim / width;
    const int x = id / non_width;
    const int num_sets = width / num_w;
    const int w_x = x / num_sets;
    d_out[id] = d_a[id] * d_w[w_x];
}

__global__ void ewsum_sum_kernel(float *d_a, float *d_out, int num_w, int width, int total_dim) {

    // out is (width / num_w) x (total_dim / width)
    const int in_set = width / num_w;
    const int non_width = total_dim / width;
    const int out_total_dim = in_set * non_width;

    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= out_total_dim) {
        return;
    }

    const int out_x = id / non_width;
    const int non_x_loc = id % non_width;

    // TODO: this is probably slow...
    float out = 0;
    for (int i = out_x; i < width; i += in_set) {
        out += d_a[i*non_width + non_x_loc];
    }
    d_out[id] = out;
}

__global__ void ewsum_back_kernel(float *d_error, float *d_w, float *d_out,
        int num_w, int err_width, int width, int total_dim) {

    // Get the id and make sure it is within bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= total_dim) {
        return;
    }

    const int non_width = total_dim / width;
    const int x = id / non_width;
    const int num_sets = width / num_w;
    const int w_x = x / num_sets;
    const int err_x = x % err_width;
    const int non_x_loc = id % non_width;
    d_out[id] = d_w[w_x] * d_error[err_x*non_width + non_x_loc];
}
