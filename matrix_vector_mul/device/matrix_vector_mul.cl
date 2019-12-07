
// Multiplies an MxN matrix by a M-dimensional vector giving an N-dimensional vector
__kernel void matrix_vector_mul(int M, int N, __global float *restrict mat, __global float *restrict vecIn, __global float *restrict vecOut) {

	for(int i = 0; i < N; i++) {
		float sum = 0.0;
		for(int j = 0; j < M; j++) {
			sum += mat[j+M*i] * vecIn[j];
		}
		vecOut[i] = sum;
	}

}