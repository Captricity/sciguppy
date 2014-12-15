// Sizes are W H D
__device__ __constant__ int d_ax_size[3];
__device__ __constant__ int d_ay_size[3];
__device__ __constant__ int d_aout_size[3];
__device__ __constant__ int d_padding[3];

// Signal correlation kernel. Only works with 3D arrays. One thread per output entry.
__global__ void correlate_kernel(float *ax, float *ay, float *aout) {

    // Extract dimensions and padding
    const int ax_w = d_ax_size[0];
    const int ax_h = d_ax_size[1];
    const int ax_d = d_ax_size[2];
    const int ay_w = d_ay_size[0];
    const int ay_h = d_ay_size[1];
    const int ay_d = d_ay_size[2];
    const int aout_w = d_aout_size[0];
    const int aout_h = d_aout_size[1];
    const int aout_d = d_aout_size[2];
    const int xpad = d_padding[0];
    const int ypad = d_padding[1];
    const int zpad = d_padding[2];
 
    // Get the id, and make sure it is not out of bounds
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= aout_w * aout_h * aout_d) {
        return;
    }

    // Now get the coordinates in the output matrix
    const int x = id / (aout_h*aout_d);
    const int y = (id % (aout_h*aout_d)) / aout_d;
    const int z = (id % (aout_h*aout_d)) % aout_d;

    int ax_id, ay_id, i, j, k;
    int ax_x, ax_y, ax_z;
    float sum = 0.0;

    // Each output entry is a reduction of matrix dot product between the two
    // inputs at that point
    for (i = 0; i < ay_w; ++i) {
        for (j = 0; j < ay_h; ++j) {
            for (k = 0; k < ay_d; ++k) {
                // Bounds check. If out of bound, ignore because ax value is 0.
                ax_x = (x+i)-xpad;
                ax_y = (y+j)-ypad;
                ax_z = (z+k)-zpad;
                if ((ax_x < 0) ||
                    (ax_y < 0) ||
                    (ax_z < 0) ||
                    (ax_x >= ax_w) ||
                    (ax_y >= ax_h) ||
                    (ax_z >= ax_d))
                    continue;

                // Correlation is sum of dot product between overlapping points
                ax_id = (ax_x * ax_h * ax_d) + (ax_y * ax_d) + ax_z;
                ay_id = (i * ay_h * ay_d) + (j * ay_d) + k;
                sum += (ax[ax_id] * ay[ay_id]);
            }
        }
    }

    aout[id] = sum;
}
