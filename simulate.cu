#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void matrixVectorMult(curandState* state, float* matrices, float* vectors, float* results, int n) {
    int matrixId = blockIdx.x;  // Each block handles one matrix and its corresponding vectors
    int vectorId = threadIdx.x; // Each thread handles one vector
    curandState localState = state[matrixId * 1000 + vectorId]; // Unique state for each vector

    // Pointers to the matrix for this block
    float* matrix = matrices + matrixId * n * 2 * n;
    float* vector = vectors + (matrixId * 1000 + vectorId) * 2 * n;
    float* result = results + (matrixId * 1000 + vectorId) * n;

    // Generate the vector
    for (int i = 0; i < 2 * n; i++) {
        vector[i] = curand_uniform(&localState);  // Fill the vector with random numbers
    }

    // Matrix-vector multiplication
    for (int row = 0; row < n; row++) {
        float sum = 0.0;
        for (int col = 0; col < 2 * n; col++) {
            sum += matrix[row * 2 * n + col] * vector[col];
        }
        result[row] = sum;
    }

    // Store the RNG state back
    state[matrixId * 1000 + vectorId] = localState;
}

// Kernel to initialize matrices
__global__ void initializeMatrices(curandState* state, float* matrices, int n) {
    int id = blockIdx.x;  // One block per matrix
    curandState localState = state[id];  // Unique state per matrix

    float* matrix = matrices + id * n * 2 * n;

    for (int i = 0; i < n * 2 * n; i++) {
        matrix[i] = curand_uniform(&localState);  // Fill the matrix with random numbers
    }

    state[id] = localState;
}

int main() {
    int n = 1024;  // dimension
    int num_matrices = 1000;
    int num_vectors_per_matrix = 1000;
    int total_vectors = num_matrices * num_vectors_per_matrix;

    // Allocate memory
    float *d_matrices, *d_vectors, *d_results;
    cudaMalloc(&d_matrices, num_matrices * n * 2 * n * sizeof(float)); //n by 2n
    cudaMalloc(&d_vectors, total_vectors * 2 * n * sizeof(float)); // 2n
    cudaMalloc(&d_results, total_vectors * n * sizeof(float)); // let's be clever and get rid of this
    curandState *d_states;
    cudaMalloc(&d_states, total_vectors * sizeof(curandState)); //vector rng

    // Initialize matrices
    int threadsPerBlock = 1;  // One thread per block for matrix initialization
    initializeMatrices<<<num_matrices, threadsPerBlock>>>(d_states, d_matrices, n);

    setup_states<<<total_vectors, 1>>>(d_states, time(NULL));  // Initialize rng states for each vector

    // Perform matrix-vector multiplications
    threadsPerBlock = 1000;  // One thread per vector, per matrix
    matrixVectorMult<<<num_matrices, threadsPerBlock>>>(d_states, d_matrices, d_vectors, d_results, n);

    // Free memory
    cudaFree(d_matrices);
    cudaFree(d_vectors);
    cudaFree(d_results);
    cudaFree(d_states);

    return 0;
}

