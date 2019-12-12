#pragma OPENCL EXTENSION cl_altera_channels : enable


/************************ COMPLEX NUMBERS ************************/

//2 component vector to hold the real and imaginary parts of a complex number:
// typedef float2 cfloat;

// #define cI ((cfloat) (0.0, 1.0))

// /*
//  * Return Real (Imaginary) component of complex number:
//  */
// inline float creal(cfloat a) {
// 	return a.x;
// }
// inline float cimag(cfloat a) {
// 	return a.y;
// }

// inline float cmod(cfloat a) {
//     return (sqrt(a.x*a.x + a.y*a.y));
// }

// inline float carg(cfloat a) {
// 	return atan2(a);
// }

// inline cfloat cadd(cfloat a, cfloat b) {
// 	return (cfloat)(a.x + b.x, a.y + b.y);
// }

// inline cfloat cmult(cfloat a, cfloat b) {
// 	return (cfloat)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
// }

// inline cfloat cdiv(cfloat a, cfloat b) {
// 	return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
// }

// inline cfloat csqrt(cfloat a) {
// 	return (cfloat)(sqrt(cmod(a)) * cos(carg(a)/2),  sqrt(cmod(a)) * sin(carg(a)/2));
// }

/************************ END COMPLEX NUMBERS ************************/


// as per Kelly (2018)
static int nthCleared(int n, int t) {
  int mask = (1 << t) - 1;
  int notMask = ~mask;

  return (n & mask) | ((n & notMask) << 1);
}

channel int metaCh;
channel float workingCh;

channel float outputCh;

// M x M gate
// N-dimensional state
__kernel void kernelInput(int M, int N, __global  float * restrict gate, __global  float * restrict state, int qID) {

	// pass M and qID to the dimensions channel
	write_channel_altera(metaCh, M);
	write_channel_altera(metaCh, N);
	write_channel_altera(metaCh, qID);

	// write the gate to the channel
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < M; j++) {
			write_channel_altera(workingCh, gate[j+M*i]);
		}
	}

	// write the state to the channel
	for(int i = 0; i < N; i++) {
		write_channel_altera(workingCh, state[i]);
	}
}

__kernel void kernelCompute() {
	const int M = read_channel_altera(metaCh);
	const int N = read_channel_altera(metaCh);
	const int qID = read_channel_altera(metaCh);

	float *gate = (float *)malloc(sizeof(float)*M*M);
	float *state = (float *)malloc(sizeof(float)*N);

	// read the gate from the channel
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < M; j++) {
			gate[j+M*i] = read_channel_altera(workingCh);
		}
	}

	// read the state from the channel
	for(int i = 0; i < N; i++) {
		state[i] = read_channel_altera(workingCh);
	}

	// N = 2^n so 2^(n-1) needed for the loop is N/2
	int loopLim = N/2;

	// multiply
	for(int i = 0; i < loopLim; i++) {
		int zero_state = nthCleared(i, qID);
		int one_state = zero_state | (1 << qID);

		 float zero_amp = state[zero_state];
		 float one_amp = state[one_state];

		state[zero_state] = gate[0]*zero_amp + gate[1]*one_amp;
		state[one_state] = gate[2]*zero_amp + gate[3]*one_amp;
	}

	// write the new state to the output channel
	for(int i = 0; i < N; i++) {
		write_channel_altera(outputCh, state[i]);
	}
}

__kernel void kernelOutput(int N, __global  float * restrict outputState) {
	for(int i = 0; i < N; i++) {
		outputState[i] = read_channel_altera(outputCh);
	}
}


