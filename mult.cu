#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%6.2f ", M[i * p + j]); 
        }
        printf("\n"); 
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i* p + j] = M1[i* p + j] + M2[i* p + j];
        }
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}





__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        Mout[row * n + col] = M1[row * n + col] + M2[row * n + col];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            Mout[row * n + col] += M1[row * n + k] * M2[k * n + col];
        }
    }

}







int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <n> <p> \n", argv[0]);
        return -1;
    }
    srand((unsigned int)time(NULL));

    int n = atoi(argv[1]);
    int p = atoi(argv[2]);
    size_t size = n * p * sizeof(float);

    // Allocate host memory
    float *M = (float *)malloc(size);
    float *M2 = (float *)malloc(size);
    float *Mout = (float *)malloc(size);

    // Allocate device memory
    float *d_M, *d_M2, *d_Mout;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_M2, size);
    cudaMalloc(&d_Mout, size);

    // Initialize host matrices
    MatrixInit(M, n, p);
    MatrixInit(M2, n, p);

    printf("M:\n");
    //MatrixPrint(M, n, p);
    printf("M2:\n");
    //MatrixPrint(M2, n, p);

    // CPU matrix addition
    clock_t begin = clock();
    MatrixAdd(M, M2, Mout, n, p);
    clock_t end = clock();
    double millisadd = (double)(end -  begin) / CLOCKS_PER_SEC;
    printf( "Finished in %f s\n", millisadd );
    printf("Mout Add (CPU):\n");
    //MatrixPrint(Mout, n, p);

    // Copy matrices to device
    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, size, cudaMemcpyHostToDevice);

    // GPU matrix addition
    dim3 blockDim(4, 4);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(Mout, d_Mout, size, cudaMemcpyDeviceToHost);
    printf("Mout Add (GPU):\n");
    //MatrixPrint(Mout, n, p);

    // CPU matrix multiplication
    clock_t begin2 = clock();
    MatrixMult(M, M2, Mout, n);
    clock_t end2 = clock();
    double millismult = (double)(end2 -  begin2) / CLOCKS_PER_SEC;
    printf( "Finished in %f s\n", millismult );
    printf("Mout Mult (CPU):\n");
    //MatrixPrint(Mout, n, n);

    // GPU matrix multiplication
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M, d_M2, d_Mout, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(Mout, d_Mout, size, cudaMemcpyDeviceToHost);
    printf("Mout Mult (GPU):\n");
    //MatrixPrint(Mout, n, n);

    // Free memory
    free(M);
    free(M2);
    free(Mout);
    cudaFree(d_M);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}