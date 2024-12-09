#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE_DATA         32
#define INPUT_SIZE_C1_DATA      28
#define INPUT_CHANNEL           6
#define INPUT_SIZE_S1           14
#define INPUT_SIZE_C1_KERNEL    5
#define POOL_SIZE               2

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
                        M[chan * n * p + i * p + j] = 0;
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
                        M[chan * n * p + i * p + j] = ((float)rand() / RAND_MAX);
                    }
                }
            }
        }
    }
        
}

void MatrixPrint(float *M, int n, int p, int c) {
    for (int chan = 0; chan < c; chan++) {
        printf("Channel %d:\n", chan);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                printf("%0.4f ", M[chan * n * p + i * p + j]); 
            }
            printf("\n"); 
        }
        printf("\n");
    }
}

__device__ float activation_tanh(float f) {
    return tanh(f);
}

__device__ void activation_softmax(float* input, float* output, int size) {
    float max_val = input[0];
    float sum = 0.0f;

    // Trouver le maximum pour stabilité numérique
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Calcul de l'exponentielle et somme
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val); // Soustraction du max pour stabilité
        sum += output[i];
    }

    // Normalisation
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
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

        output[kernel_idx * output_size * output_size + row * output_size + col] = activation_tanh(sum);
    }
}



__global__ void avg_pooling(const float* input, float* output, int input_size, int output_size, int pool_size, int num_channels) {
    int kernel_idx = blockIdx.z; // Index du canal
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index de la ligne dans l'output
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index de la colonne dans l'output

    // Vérification pour éviter les débordements
    if (kernel_idx < num_channels && row < output_size && col < output_size) {
        float sum = 0.0f;

        // Accumulation des valeurs dans la fenêtre de pooling
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                int r = row * pool_size + i; // Position dans l'input
                int c = col * pool_size + j; // Position dans l'input
                sum += input[kernel_idx * input_size * input_size + r * input_size + c];
            }
        }

        // Calcul de la moyenne et écriture dans l'output
        output[kernel_idx * output_size * output_size + row * output_size + col] = sum / (pool_size * pool_size);
    }
}








int main() {
    srand(time(NULL));

    // Allocation de la mémoire pour les matrices CPU
    float *raw_data = (float *)malloc(INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float));
    float *C1_data = (float *)malloc(INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float));
    float *S2_data = (float *)malloc(INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float));
    float *C3_data = (float *)malloc(16 * (INPUT_SIZE_S1 - 5 + 1) * (INPUT_SIZE_S1 - 5 + 1) * sizeof(float));
    float *S4_data = (float *)malloc(16 * (INPUT_SIZE_S1 / 2 - 2 + 1) * (INPUT_SIZE_S1 / 2 - 2 + 1) * sizeof(float));

    float *C1_kernel = (float *)malloc(INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));
    float *C3_kernel = (float *)malloc(16 * INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));

    // Initialisation des données
    MatrixInit(raw_data, INPUT_SIZE_DATA, INPUT_SIZE_DATA, 1, false);
    MatrixInit(C1_kernel, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_KERNEL, INPUT_CHANNEL, false);
    MatrixInit(C3_kernel, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_KERNEL, 16, false);

    // Pointeurs GPU
    float *d_raw_data, *d_C1_data, *d_S2_data, *d_C1_kernel, *d_C3_data, *d_S4_data, *d_C3_kernel;
    cudaMalloc((void **)&d_raw_data, INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float));
    cudaMalloc((void **)&d_C1_data, INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float));
    cudaMalloc((void **)&d_S2_data, INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));
    cudaMalloc((void **)&d_C3_data, 16 * (INPUT_SIZE_S1 - 5 + 1) * (INPUT_SIZE_S1 - 5 + 1) * sizeof(float));
    cudaMalloc((void **)&d_S4_data, 16 * (INPUT_SIZE_S1 / 2 - 2 + 1) * (INPUT_SIZE_S1 / 2 - 2 + 1) * sizeof(float));
    cudaMalloc((void **)&d_C3_kernel, 16 * INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));

    // Copie des données sur le GPU
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_kernel, C3_kernel, 16 * INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float), cudaMemcpyHostToDevice);

    // C1: Convolution
    dim3 blockDim(16, 16);
    dim3 gridDim((INPUT_SIZE_C1_DATA + blockDim.x - 1) / blockDim.x, (INPUT_SIZE_C1_DATA + blockDim.y - 1) / blockDim.y, INPUT_CHANNEL);
    conv2d<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data, INPUT_SIZE_DATA, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_DATA, INPUT_CHANNEL);
    cudaMemcpy(C1_data, d_C1_data, INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nRésultat après la convolution (C1) avec activation tanh :\n");
    MatrixPrint(C1_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_C1_DATA, INPUT_CHANNEL);

    // S2: Pooling moyen
    avg_pooling<<<gridDim, blockDim>>>(d_C1_data, d_S2_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_S1, POOL_SIZE, INPUT_CHANNEL);
    cudaMemcpy(S2_data, d_S2_data, INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nRésultat après pooling moyen (S2) :\n");
    MatrixPrint(S2_data, INPUT_SIZE_S1, INPUT_SIZE_S1, INPUT_CHANNEL);

    // C3: Convolution + tanh
    dim3 gridDimC3((INPUT_SIZE_S1 - 5 + 1 + blockDim.x - 1) / blockDim.x, (INPUT_SIZE_S1 - 5 + 1 + blockDim.y - 1) / blockDim.y, 16);
    conv2d<<<gridDimC3, blockDim>>>(d_S2_data, d_C3_kernel, d_C3_data, INPUT_SIZE_S1, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_S1 - 5 + 1, 16);
    cudaMemcpy(C3_data, d_C3_data, 16 * (INPUT_SIZE_S1 - 5 + 1) * (INPUT_SIZE_S1 - 5 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nRésultat après la convolution (C3) avec activation tanh :\n");
    MatrixPrint(C3_data, INPUT_SIZE_S1 - 5 + 1, INPUT_SIZE_S1 - 5 + 1, 16);

    // S4: Pooling moyen
    avg_pooling<<<gridDimC3, blockDim>>>(d_C3_data, d_S4_data, INPUT_SIZE_S1 - 5 + 1, INPUT_SIZE_S1 / 2 - 2 + 1, POOL_SIZE, 16);
    cudaMemcpy(S4_data, d_S4_data, 16 * (INPUT_SIZE_S1 / 2 - 2 + 1) * (INPUT_SIZE_S1 / 2 - 2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nRésultat après pooling moyen (S4) :\n");
    MatrixPrint(S4_data, INPUT_SIZE_S1 / 2 - 2 + 1, INPUT_SIZE_S1 / 2 - 2 + 1, 16);


    // Libération de la mémoire
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S2_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C3_data);
    cudaFree(d_S4_data);
    cudaFree(d_C3_kernel);

    free(raw_data);
    free(C1_data);
    free(S2_data);
    free(C3_data);
    free(S4_data);
    free(C1_kernel);
    free(C3_kernel);

    return 0;
}
