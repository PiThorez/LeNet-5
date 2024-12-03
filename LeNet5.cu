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
   float* raw_data = (float*)malloc(INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float));
   float* C1_data = (float*)malloc(INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float));
   float* S1_data = (float*)malloc(INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float));
   float* C1_kernel = (float*)malloc(INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));

   // Initialisation des matrices
   MatrixInit(raw_data, INPUT_SIZE_DATA, INPUT_SIZE_DATA, 1, false);  // Valeurs aléatoires entre 0 et 1 pour raw_data
   MatrixInit(C1_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_C1_DATA, INPUT_CHANNEL, true);  // Initialisation à zéro pour C1_data
   MatrixInit(S1_data, INPUT_SIZE_S1, INPUT_SIZE_S1, INPUT_CHANNEL, true);  // Initialisation à zéro pour S1_data
   MatrixInit(C1_kernel, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_KERNEL, INPUT_CHANNEL, false);  // Valeurs aléatoires pour C1_kernel

   // Pointeurs GPU
   float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;

   // Allocation mémoire sur le GPU
   cudaMalloc((void**)&d_raw_data, INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float));
   cudaMalloc((void**)&d_C1_data, INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float));
   cudaMalloc((void**)&d_S1_data, INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float));
   cudaMalloc((void**)&d_C1_kernel, INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float));

   // Copie des données vers le GPU
   cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE_DATA * INPUT_SIZE_DATA * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_C1_kernel, C1_kernel, INPUT_CHANNEL * INPUT_SIZE_C1_KERNEL * INPUT_SIZE_C1_KERNEL * sizeof(float), cudaMemcpyHostToDevice);

   // Définition des dimensions des blocs et des grilles
   dim3 blockDim(16, 16);
   dim3 gridDim((INPUT_SIZE_C1_DATA + blockDim.x - 1) / blockDim.x, (INPUT_SIZE_C1_DATA + blockDim.y - 1) / blockDim.y, INPUT_CHANNEL);

   // Lancer la convolution avec activation
   conv2d<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data, INPUT_SIZE_DATA, INPUT_SIZE_C1_KERNEL, INPUT_SIZE_C1_DATA, INPUT_CHANNEL);
   cudaGetLastError(), "cudaConvolution2DWithActivation";

   // Copier les résultats de la convolution
   cudaMemcpy(C1_data, d_C1_data, INPUT_CHANNEL * INPUT_SIZE_C1_DATA * INPUT_SIZE_C1_DATA * sizeof(float), cudaMemcpyDeviceToHost);

   // Affichage des résultats après convolution et activation
   printf("\nRésultat de la convolution avec activation tanh (C1_data) :\n");
   MatrixPrint(C1_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_C1_DATA, INPUT_CHANNEL);

   // Lancer le pooling (average pooling)
   dim3 poolGridDim((INPUT_SIZE_S1 + blockDim.x - 1) / blockDim.x, (INPUT_SIZE_S1 + blockDim.y - 1) / blockDim.y, INPUT_CHANNEL);
   avg_pooling<<<poolGridDim, blockDim>>>(d_C1_data, d_S1_data, INPUT_SIZE_C1_DATA, INPUT_SIZE_S1, POOL_SIZE, INPUT_CHANNEL);
   cudaGetLastError(), "cudaAveragePooling";

   // Copier les résultats du pooling
   cudaMemcpy(S1_data, d_S1_data, INPUT_CHANNEL * INPUT_SIZE_S1 * INPUT_SIZE_S1 * sizeof(float), cudaMemcpyDeviceToHost);

   // Affichage des résultats après pooling
   printf("\nRésultat après pooling (S1_data) :\n");
   MatrixPrint(S1_data, INPUT_SIZE_S1, INPUT_SIZE_S1, INPUT_CHANNEL);

   // Libération de la mémoire
   cudaFree(d_raw_data);
   cudaFree(d_C1_data);
   cudaFree(d_S1_data);
   cudaFree(d_C1_kernel);

   free(raw_data);
   free(C1_data);
   free(S1_data);
   free(C1_kernel);

   return 0;
}