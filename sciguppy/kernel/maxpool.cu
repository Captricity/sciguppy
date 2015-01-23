// Sizes are D H W
__device__ __constant__ int d_a_size[3];
__device__ __constant__ int d_out_size[3];

__global__ void maxpool_kernel(float *d_a, float *d_out, int *d_out_idxs, int h, int w) {
    __shared__ float shared[512]; // Max allowed

    const int y = threadIdx.x;
    const int x = threadIdx.y;
    const int block_z = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_x = blockIdx.z;
    const int is_champion = x == 0 && y == 0;

    const int a_d = d_a_size[0];
    const int a_h = d_a_size[1];
    const int a_w = d_a_size[2];
    const int out_d = d_out_size[0];
    const int out_h = d_out_size[1];
    const int out_w = d_out_size[2];

    const int a_z = block_z;
    const int a_y = block_y*h + y;
    const int a_x = block_x*w + x;
    const int a_idx = (a_z*a_h*a_w) + (a_y*a_w) + a_x;

    const int shared_idx = (y*w) + x;

    const int out_z = a_z;
    const int out_y = block_y;
    const int out_x = block_x;
    const int out_idx = (out_z*out_h*out_w) + (out_y*out_w) + out_x;
    const int out_idxs_idx_base = (out_z*out_h*out_w*2) + (out_y*out_w*2) + (out_x*2);

    if (a_z >= a_d || a_y >= a_h || a_x >= a_w) {
        return;
    }
    if (out_z >= out_d || out_y >= out_h || out_x >= out_w) {
        return;
    }

    shared[shared_idx] = d_a[a_idx];
    __syncthreads();

    if (is_champion) {
        // Pulled from math_constants.h
        float max_val = __int_as_float(0xff800000);
        int max_idx;
        for (int i = 0; i < h*w; ++i) {
            if (shared[i] > max_val) {
                max_val = shared[i];
                max_idx = i;
            }
        }
        d_out[out_idx] = max_val;
        const int max_idx_y = max_idx / w;
        const int max_idx_x = max_idx % w;
        d_out_idxs[out_idxs_idx_base+0] = block_y*h + max_idx_y;
        d_out_idxs[out_idxs_idx_base+1] = block_x*w + max_idx_x;
    }
}

__global__ void maxpool_back_kernel(float *d_error, int *d_max_idxs, float *d_out) {
    //const int out_d = d_out_size[0]; // Not used
    const int out_h = d_out_size[1];
    const int out_w = d_out_size[2];
    const int error_d = d_a_size[0];
    const int error_h = d_a_size[1];
    const int error_w = d_a_size[2];

    // Get the id, and make sure it is not out of bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= error_d*error_h*error_w) {
        return;
    }

    // Now get the coordinates in the max_idxs matrix
    const int z = id / (error_h*error_w);
    const int y = (id % (error_h*error_w)) / error_w;
    const int x = (id % (error_h*error_w)) % error_w;
    const int max_idxs_base_idx = (z*error_h*error_w*2) + (y*error_w*2) + (x*2);
    
    // ... and extract the output coordinates from the max_idxs matrix
    const int out_y = d_max_idxs[max_idxs_base_idx+0];
    const int out_x = d_max_idxs[max_idxs_base_idx+1];
    const int out_idx = (z*out_h*out_w) + (out_y*out_w) + out_x;

    // Now assign the error
    d_out[out_idx] = d_error[id];
}
