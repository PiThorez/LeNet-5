#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define INPUT_SIZE 32          // Taille de l'entrée 32x32
#define C1_KERNEL_SIZE 5       // Taille des noyaux de convolution 5x5
#define C1_OUTPUT_SIZE 28      // Taille après convolution
#define S1_OUTPUT_SIZE 14      // Taille après pooling
#define NUM_KERNELS_C1 6       // Nombre de noyaux de C1
#define C2_KERNEL_SIZE 5       // Taille des noyaux de C2
#define C2_OUTPUT_SIZE 10      // Taille après C2
#define NUM_KERNELS_C2 16      // Nombre de noyaux de C2
#define S2_OUTPUT_SIZE 5       // Taille après pooling
#define FC1_OUTPUT_SIZE 120    // Neurones de la couche fully connected 1
#define FC2_OUTPUT_SIZE 84     // Neurones de la couche fully connected 2
#define NUM_CLASSES 10         // Nombre de classes (couche de sortie)
#define POOL_SIZE 2            // Taille du pool (2x2)
#define STRIDE 2               // Stride pour le pooling

void initializeRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;  // Valeur aléatoire entre 0 et 1
    }
}

// Fonction pour initialiser une matrice à 0
void initializeZeroMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f;
    }
}

// Fonction d'activation tanh pour GPU
__device__ float activation_tanh(float x) {
    return tanh(x);
}

// Fonction d'activation softmax pour la couche de sortie
__device__ float activation_softmax(float x, float* output, int size, int idx) {
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(output[i]);
    }
    return expf(x) / sum_exp;
}

// Kernel de convolution avec activation tanh
__global__ void cudaConvolution2DWithActivation(float* input, float* output, float* kernel, int input_size, int kernel_size, int output_size, int num_kernels) {
    int k = blockIdx.z;  // Indice du noyau (pour chaque noyau)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice de ligne
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice de colonne

    if (i < output_size && j < output_size) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_size; m++) {
            for (int n = 0; n < kernel_size; n++) {
                int x = i + m;
                int y = j + n;
                if (x < input_size && y < input_size) {
                    sum += input[x * input_size + y] * kernel[k * kernel_size * kernel_size + m * kernel_size + n];
                }
            }
        }
        // Activation tanh
        output[k * output_size * output_size + i * output_size + j] = activation_tanh(sum);
    }
}

// Kernel de pooling (average pooling)
__global__ void cudaAveragePooling(float* input, float* output, int input_size, int output_size, int num_kernels, int pool_size, int stride) {
    int k = blockIdx.z;  // Indice du noyau (pour chaque noyau)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice de ligne
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice de colonne

    if (i < output_size && j < output_size) {
        float sum = 0.0f;
        for (int m = 0; m < pool_size; m++) {
            for (int n = 0; n < pool_size; n++) {
                int x = i * stride + m;
                int y = j * stride + n;

                if (x < input_size && y < input_size) {
                    sum += input[k * input_size * input_size + x * input_size + y];
                }
            }
        }
        output[k * output_size * output_size + i * output_size + j] = sum / (pool_size * pool_size); // Moyenne pour average pooling
    }
}

// Fonction pour aplatissement (Flatten) de la sortie
__global__ void flatten(float* input, float* output, int input_size, int num_kernels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * input_size * num_kernels) {
        output[idx] = input[idx];
    }
}

// Fonction pour la couche dense
__global__ void denseLayer(float* input, float* output, float* weights, float* biases, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = biases[idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = activation_tanh(sum);
    }
}

// Fonction pour la couche de sortie (softmax)
__global__ void outputLayer(float* input, float* output, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        output[idx] = activation_softmax(input[idx], output, output_size, idx);
    }
}

int main() {
    srand(time(NULL));

    // Allocation mémoire pour les matrices CPU
    float* raw_data = (float*)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(float));
    float* C1_data = (float*)malloc(NUM_KERNELS_C1 * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float));
    float* S1_data = (float*)malloc(NUM_KERNELS_C1 * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE * sizeof(float));
    float* C1_kernel = (float*)malloc(NUM_KERNELS_C1 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    float* C2_data = (float*)malloc(NUM_KERNELS_C2 * C2_OUTPUT_SIZE * C2_OUTPUT_SIZE * sizeof(float));
    float* S2_data = (float*)malloc(NUM_KERNELS_C2 * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * sizeof(float));
    float* C2_kernel = (float*)malloc(NUM_KERNELS_C2 * C2_KERNEL_SIZE * C2_KERNEL_SIZE * sizeof(float));
    
    // Couches fully connected
    float* fc1_weights = (float*)malloc(FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float));
    float* fc1_biases = (float*)malloc(FC1_OUTPUT_SIZE * sizeof(float));
    float* fc2_weights = (float*)malloc(FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float));
    float* fc2_biases = (float*)malloc(FC2_OUTPUT_SIZE * sizeof(float));
    float* output_weights = (float*)malloc(NUM_CLASSES * FC2_OUTPUT_SIZE * sizeof(float));
    float* output_biases = (float*)malloc(NUM_CLASSES * sizeof(float));

    // Initialisation des matrices
    initializeRandomMatrix(raw_data, INPUT_SIZE * INPUT_SIZE);  // Valeurs aléatoires pour raw_data
    initializeZeroMatrix(C1_data, NUM_KERNELS_C1 * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE);  // Initialisation à zéro pour C1_data
    initializeZeroMatrix(S1_data, NUM_KERNELS_C1 * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE);  // Initialisation à zéro pour S1_data
    initializeRandomMatrix(C1_kernel, NUM_KERNELS_C1 * C1_KERNEL_SIZE * C1_KERNEL_SIZE);  // Valeurs aléatoires pour C1_kernel
    initializeRandomMatrix(C2_kernel, NUM_KERNELS_C2 * C2_KERNEL_SIZE * C2_KERNEL_SIZE);  // Valeurs aléatoires pour C2_kernel
    initializeRandomMatrix(fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2);  // Initialisation aléatoire des poids FC1
    initializeRandomMatrix(fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE);  // Initialisation aléatoire des poids FC2
    initializeRandomMatrix(output_weights, NUM_CLASSES * FC2_OUTPUT_SIZE);  // Initialisation aléatoire des poids de sortie

    // Allocation mémoire sur le GPU
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel, *d_C2_kernel, *d_C2_data, *d_S2_data;
    float *d_flattened_data, *d_fc1_output, *d_fc2_output, *d_output;
    float *d_fc1_weights, *d_fc1_biases, *d_fc2_weights, *d_fc2_biases, *d_output_weights, *d_output_biases;
    cudaMalloc((void**)&d_raw_data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C1_data, NUM_KERNELS_C1 * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_S1_data, NUM_KERNELS_C1 * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C1_kernel, NUM_KERNELS_C1 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C2_data, NUM_KERNELS_C2 * C2_OUTPUT_SIZE * C2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_S2_data, NUM_KERNELS_C2 * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C2_kernel, NUM_KERNELS_C2 * C2_KERNEL_SIZE * C2_KERNEL_SIZE * sizeof(float));

    cudaMalloc((void**)&d_fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float));
    cudaMalloc((void**)&d_fc1_biases, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_fc2_biases, FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output_weights, NUM_CLASSES * FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output_biases, NUM_CLASSES * sizeof(float));

    // Allocation pour le flatten
    cudaMalloc((void**)&d_flattened_data, S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float));

    // Allocation pour les couches fully connected
    cudaMalloc((void**)&d_fc1_output, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_fc2_output, FC2_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, NUM_CLASSES * sizeof(float));

    // Copier les données vers le GPU
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS_C1 * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_kernel, C2_kernel, NUM_KERNELS_C2 * C2_KERNEL_SIZE * C2_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weights, fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, fc1_biases, FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weights, fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases, fc2_biases, FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, output_weights, NUM_CLASSES * FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_biases, output_biases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);

    // Définition des dimensions des blocs et des grilles
    dim3 blockDim(16, 16);
    dim3 gridDim_C1((C1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (C1_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS_C1);
    
    // Convolution C1
    cudaConvolution2DWithActivation<<<gridDim_C1, blockDim>>>(d_raw_data, d_C1_data, d_C1_kernel, INPUT_SIZE, C1_KERNEL_SIZE, C1_OUTPUT_SIZE, NUM_KERNELS_C1);

    // Pooling S1
    dim3 gridDim_S1((S1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (S1_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS_C1);
    cudaAveragePooling<<<gridDim_S1, blockDim>>>(d_C1_data, d_S1_data, C1_OUTPUT_SIZE, S1_OUTPUT_SIZE, NUM_KERNELS_C1, POOL_SIZE, STRIDE);

    // Convolution C2
    dim3 gridDim_C2((C2_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (C2_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS_C2);
    cudaConvolution2DWithActivation<<<gridDim_C2, blockDim>>>(d_S1_data, d_C2_data, d_C2_kernel, S1_OUTPUT_SIZE, C2_KERNEL_SIZE, C2_OUTPUT_SIZE, NUM_KERNELS_C2);

    // Pooling S2
    dim3 gridDim_S2((S2_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (S2_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS_C2);
    cudaAveragePooling<<<gridDim_S2, blockDim>>>(d_C2_data, d_S2_data, C2_OUTPUT_SIZE, S2_OUTPUT_SIZE, NUM_KERNELS_C2, POOL_SIZE, STRIDE);

    // Flattening
    cudaMemcpy(d_flattened_data, d_S2_data, S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // FC1 (Première couche fully connected)
    dim3 gridDim_FC1((FC1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 1, 1);
    cudaMemcpy(d_fc1_weights, fc1_weights, FC1_OUTPUT_SIZE * S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, fc1_biases, FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    denseLayer<<<gridDim_FC1, blockDim>>>(d_flattened_data, d_fc1_output, d_fc1_weights, d_fc1_biases, S2_OUTPUT_SIZE * S2_OUTPUT_SIZE * NUM_KERNELS_C2, FC1_OUTPUT_SIZE);

    // FC2 (Deuxième couche fully connected)
    dim3 gridDim_FC2((FC2_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, 1, 1);
    cudaMemcpy(d_fc2_weights, fc2_weights, FC2_OUTPUT_SIZE * FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases, fc2_biases, FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    denseLayer<<<gridDim_FC2, blockDim>>>(d_fc1_output, d_fc2_output, d_fc2_weights, d_fc2_biases, FC1_OUTPUT_SIZE, FC2_OUTPUT_SIZE);

    // Sortie (softmax)
    dim3 gridDim_output((NUM_CLASSES + blockDim.x - 1) / blockDim.x, 1, 1);
    cudaMemcpy(d_output_weights, output_weights, NUM_CLASSES * FC2_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_biases, output_biases, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice);
    outputLayer<<<gridDim_output, blockDim>>>(d_fc2_output, d_output, NUM_CLASSES);

    // Récupérer les résultats sur le CPU
    float* output_result = (float*)malloc(NUM_CLASSES * sizeof(float));
    cudaMemcpy(output_result, d_output, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    // Afficher les résultats (classe prédit)
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Classe %d: %f\n", i, output_result[i]);
    }

    // Libérer la mémoire
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(C2_data);
    free(S2_data);
    free(C2_kernel);
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    free(output_weights);
    free(output_biases);
    free(output_result);

    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C2_data);
    cudaFree(d_S2_data);
    cudaFree(d_C2_kernel);
    cudaFree(d_flattened_data);
    cudaFree(d_fc1_output);
    cudaFree(d_fc2_output);
    cudaFree(d_output);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_biases);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc2_biases);
    cudaFree(d_output_weights);
    cudaFree(d_output_biases);


    return 0;
}