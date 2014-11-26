// Sizes are W H D
__device__ __constant__ int d_ax_size[3];
__device__ __constant__ int d_ay_size[3];
__device__ __constant__ int d_aout_size[3];

__global__ void correlate_kernel(float *ax, float *ay, float *aout) {

    //const int ax_w = d_ax_size[0]; // Unused
    const int ax_h = d_ax_size[1];
    const int ax_d = d_ax_size[2];
    const int ay_w = d_ay_size[0];
    const int ay_h = d_ay_size[1];
    const int ay_d = d_ay_size[2];
    const int aout_w = d_aout_size[0];
    const int aout_h = d_aout_size[1];
    const int aout_d = d_aout_size[2];
 
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= aout_w * aout_h * aout_d) {
        return;
    }

    const int x = id / (aout_h*aout_d);
    const int y = (id % (aout_h*aout_d)) / aout_d;
    const int z = (id % (aout_h*aout_d)) % aout_d;
    int ax_id, ay_id, i, j, k;
    float sum = 0.0;

    for (i = 0; i < ay_w; ++i) {
        for (j = 0; j < ay_h; ++j) {
            for (k = 0; k < ay_d; ++k) {
                ax_id = ((x+i) * ax_h * ax_d) + ((y+j) * ax_d) + (z+k);
                ay_id = (i * ay_h * ay_d) + (j * ay_d) + k;
                sum += (ax[ax_id] * ay[ay_id]);
            }
        }
    }

    aout[id] = sum;
}
