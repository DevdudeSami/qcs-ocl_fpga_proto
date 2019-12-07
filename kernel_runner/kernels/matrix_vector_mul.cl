
// Multiplies an MxN matrix by a M-dimensional vector giving an N-dimensional vector
__kernel void matrix_vector_mul(float *M, float *N, __global float *restrict mat, __global float *restrict vecIn, __global float *restrict vecOut) {

	float m = *M;
	float n = *N;

	for(int i = 0; i < n; i++) {
		float sum = 0.0;
		for(int j = 0; j < m; j++) {
			sum += mat[j+m*i] * vecIn[j];
		}
		vecOut[i] = sum;
	}

}