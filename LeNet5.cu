#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE_DATA         32
#define INPUT_SIZE_C1_DATA      28
#define INPUT_CHANNEL           6
#define INPUT_SIZE_S1           14
#define INPUT_SIZE_C1_KERNEL    5

void MatrixInit(float *M, int n, int p, int c, bool is_zeros) {
    if (is_zeros) {
        if (c==1) {    
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    M[i * p + j] = 0;
                }
            }
        } else {
            for (int chan = 0; chan < c; chan++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < p; j++) {
                        M[chan * n + i * p + j] = 0;
                    }
                }
            }
        }
    } else {
        if (c==1) {    
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    M[i * p + j] = ((float)rand() / RAND_MAX);
                }
            }
        } else {
            for (int chan = 0; chan < c; chan++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < p; j++) {
                        M[chan * n + i * p + j] = ((float)rand() / RAND_MAX);
                    }
                }
            }
        }
    }
        
}

void MatrixPrint(float *M, int n, int p, int c) {
    for (int chan = 0; chan < c; chan++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                printf("%6.2f ", M[i * p + j]); 
            }
            printf("\n"); 
        }
        printf("\n");
    }
        
}

__global__ void conv2d(float* input, float* kernels, float* output, int input_size, int kernel_size, int output_size, int num_kernels) {
    int kernel_idx = blockIdx.z;                       
    int row = blockIdx.y * blockDim.y + threadIdx.y;    
    int col = blockIdx.x * blockDim.x + threadIdx.x;   

    if (row < output_size && col < output_size) {
        float sum = 0.0f;

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int r = row + i;
                int c = col + j;
                sum += input[r * input_size + c] * kernels[kernel_idx * kernel_size * kernel_size + i * kernel_size + j];
            }
        }

        output[kernel_idx * output_size * output_size + row * output_size + col] = sum;
    }
}

__global__ void avg_pooling(const float* input, float* output, int input_size, int output_size, int pool_size) {
    int kernel_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_size && col < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int r = row * pool_size + i;
                int c = col * pool_size + j;
                sum += input[kernel_idx * input_size * input_size + r * input_size + c];
            }
        }
        output[kernel_idx * output_size * output_size + row * output_size + col] = sum / (pool_size * pool_size);
    }
}



int main(void) {
    int size_data = INPUT_SIZE_DATA*INPUT_SIZE_DATA*sizeof(float);                              //Size of raw_data
    int size_C1_data = INPUT_SIZE_C1_DATA*INPUT_SIZE_C1_DATA*INPUT_CHANNEL*sizeof(float);       //Size of C1
    int size_S1_data = INPUT_SIZE_S1*INPUT_SIZE_S1*INPUT_CHANNEL*sizeof(float);                 //Size of S1
    int size_C1_kern = INPUT_SIZE_C1_KERNEL*INPUT_SIZE_C1_KERNEL*INPUT_CHANNEL*sizeof(float);   //Size of C1 kernel

    float* raw_data = (float *)malloc(size_data);                                               //Creation of raw_data
    float* C1_data = (float *)malloc(size_C1_data);                                             //Creation of C1
    float* S1_data = (float *)malloc(size_S1_data);                                             //Creation of S1
    float* C1_kernel = (float *)malloc(size_C1_kern);                                           //Creation of C1 kernel

    MatrixInit(raw_data, INPUT_SIZE_DATA, INPUT_SIZE_DATA, 1, false);
    MatrixInit(C1_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_C1_DATA, INPUT_CHANNEL, true);
    MatrixInit(S1_data, INPUT_SIZE_S1, INPUT_SIZE_S1, INPUT_CHANNEL, true);
    MatrixInit(C1_kernel, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_KERNEL, INPUT_CHANNEL, false);

    float *d_input, *d_kernels, *d_output;
    cudaMalloc(&d_input, size_data);
    cudaMalloc(&d_kernels, size_C1_kern);
    cudaMalloc(&d_output, size_C1_data);

    cudaMemcpy(d_input, raw_data, size_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, C1_kernel, size_C1_kern, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((INPUT_SIZE_C1_DATA + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (INPUT_SIZE_C1_DATA + threadsPerBlock.y - 1) / threadsPerBlock.y,
                INPUT_CHANNEL);

    conv2d<<<numBlocks, threadsPerBlock>>>(d_input, d_kernels, d_output, INPUT_SIZE_DATA, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_DATA, INPUT_CHANNEL);
    cudaDeviceSynchronize();

    float* h_output = (float*)malloc(size_C1_data);
    cudaMemcpy(h_output, d_output, size_C1_data, cudaMemcpyDeviceToHost);

    MatrixPrint(h_output, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_DATA, INPUT_SIZE_C1_DATA);

    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_output);

    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}