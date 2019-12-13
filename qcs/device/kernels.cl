#pragma OPENCL EXTENSION cl_altera_channels : enable


/************************ COMPLEX NUMBERS ************************/

//2 component vector to hold the real and imaginary parts of a complex number:
typedef float2 cfloat;

#define cI ((cfloat) (0.0, 1.0))

/*
 * Return Real (Imaginary) component of complex number:
 */
inline float creal(cfloat a) {
	return a.x;
}
inline float cimag(cfloat a) {
	return a.y;
}

inline float cmod(cfloat a) {
    return (sqrt(a.x*a.x + a.y*a.y));
}

inline float carg(cfloat a) {
	return atan2(a.y, a.x);
}

inline cfloat cadd(cfloat a, cfloat b) {
	return (cfloat)(a.x + b.x, a.y + b.y);
}

inline cfloat cmult(cfloat a, cfloat b) {
	return (cfloat)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline cfloat cdiv(cfloat a, cfloat b) {
	return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}

inline cfloat csqrt(cfloat a) {
	return (cfloat)(sqrt(cmod(a)) * cos(carg(a)/2),  sqrt(cmod(a)) * sin(carg(a)/2));
}

/************************ END COMPLEX NUMBERS ************************/


// as per Kelly (2018)
static int nthCleared(int n, int t) {
  int mask = (1 << t) - 1;
  int notMask = ~mask;

  return (n & mask) | ((n & notMask) << 1);
}

channel int metaCh;
channel float realCh;
channel float imagCh;

channel float realOutCh;
channel float imagOutCh;

// M x M gate
// N-dimensional state
__kernel void kernelInput(int G, int n, __global int * restrict qTargetsCount, __global int * restrict qTargets, __global  float * restrict gates_r, __global  float * restrict gates_i, __global  float * restrict state_r, __global  float * restrict state_i) {

	// sum qTargetCounts
	// and total gates size
	int totalTargetCounts = 0;
	int totalGatesLength = 0;
	for(int i = 0; i < G; i++) {
		totalTargetCounts += qTargetsCount[i];
		totalGatesLength += pow(pow(2,qTargetsCount[i]),2);
	}
	int N = pow(2,n);

	// pass to the meta data channel
	write_channel_altera(metaCh, G);
	write_channel_altera(metaCh, n);
	write_channel_altera(metaCh, totalTargetCounts);
	write_channel_altera(metaCh, totalGatesLength);
	for(int i = 0; i < G; i++) write_channel_altera(metaCh, qTargetsCount[i]);
	for(int i = 0; i < totalTargetCounts; i++) write_channel_altera(metaCh, qTargets[i]);

	// write the gate to the channel
	for(int i = 0; i < totalGatesLength; i++) {
		write_channel_altera(realCh, gates_r[i]);
		write_channel_altera(imagCh, gates_i[i]);
	}

	// write the state to the channel
	for(int i = 0; i < N; i++) {
		write_channel_altera(realCh, state_r[i]);
		write_channel_altera(imagCh, state_i[i]);
	}
}

__kernel void kernelCompute() {
	const int G = read_channel_altera(metaCh);
	const int n = read_channel_altera(metaCh);
	const int totalTargetCounts = read_channel_altera(metaCh);
	const int totalGatesLength = read_channel_altera(metaCh);

	int *qTargetsCount = (int *)malloc(sizeof(int)*G);
	for(int i = 0; i < G; i++) qTargetsCount[i] = read_channel_altera(metaCh);

	int *qTargets = (int *)malloc(sizeof(int)*totalTargetCounts);
	for(int i = 0; i < totalTargetCounts; i++) qTargets[i] = read_channel_altera(metaCh);

	cfloat *gates = (cfloat *)malloc(sizeof(cfloat)*totalGatesLength);
	for(int i = 0; i < totalGatesLength; i++) gates[i] = (cfloat)(read_channel_altera(realCh), read_channel_altera(imagCh));

	const int N = pow(2,n);
	cfloat *state = (cfloat *)malloc(sizeof(cfloat)*N);
	for(int i = 0; i < N; i++) state[i] = (cfloat)(read_channel_altera(realCh), read_channel_altera(imagCh));

	// N = 2^n so 2^(n-1) needed for the loop is N/2
	int loopLim = N/2;

	// consume qTargets as a tape
	int qIDTapeHead = 0;
	int gateTapeHead = 0;
	for(int g = 0; g < G; g++) {
		int targetCount = qTargetsCount[g];
		int *qIDs = (int *)malloc(sizeof(int)*targetCount);
		for(int i = 0; i < targetCount; i++) {
			qIDs[i] = qTargets[qIDTapeHead++];
		}

		int targetN = pow(pow(2, targetCount), 2);
		cfloat *gate = (cfloat *)malloc(sizeof(cfloat)*targetN);
		for(int i = 0; i < targetN; i++) {
			gate[i] = gates[gateTapeHead++];
		}

		// multiply
		for(int i = 0; i < loopLim; i++) {
			int qID = qIDs[0];

			int zero_state = nthCleared(i, qID);
			int one_state = zero_state | (1 << qID);

			cfloat zero_amp = state[zero_state];
			cfloat one_amp = state[one_state];

			state[zero_state] = cadd(cmult(gate[0],zero_amp), cmult(gate[1],one_amp));
			state[one_state] = cadd(cmult(gate[2],zero_amp), cmult(gate[3],one_amp));
		}

	}

	// multiply
	// for(int i = 0; i < loopLim; i++) {
	// 	int zero_state = nthCleared(i, qID);
	// 	int one_state = zero_state | (1 << qID);

	// 	cfloat zero_amp = state[zero_state];
	// 	cfloat one_amp = state[one_state];

	// 	state[zero_state] = cadd(cmult(gate[0],zero_amp), cmult(gate[1],one_amp));
	// 	state[one_state] = cadd(cmult(gate[2],zero_amp), cmult(gate[3],one_amp));
	// }

	// write the new state to the output channel
	for(int i = 0; i < N; i++) {
		write_channel_altera(realOutCh, creal(state[i]));
		write_channel_altera(imagOutCh, cimag(state[i]));
	}
}

__kernel void kernelOutput(int N, __global  float * restrict state_real, __global float * restrict state_imag) {
	for(int i = 0; i < N; i++) {
		state_real[i] = read_channel_altera(realOutCh);
		state_imag[i] = read_channel_altera(imagOutCh);
	}
}


