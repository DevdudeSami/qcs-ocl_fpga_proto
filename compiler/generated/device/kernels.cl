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

typedef struct cvec2_ {
	cfloat a;
	cfloat b;
} cvec2;

// // /****** GATES ******/
// constant cfloat H[] = { 
// 	(cfloat)(M_SQRT1_2_F, 0), (cfloat)(M_SQRT1_2_F, 0), 
// 	(cfloat)(M_SQRT1_2_F, 0), (cfloat)(-M_SQRT1_2_F, 0) 
// }; // 0
// constant cfloat CNOT[] = { 
// 	(cfloat)(1, 0), (cfloat)(0, 0), (cfloat)(0,0), (cfloat)(0,0),
// 	(cfloat)(0, 0), (cfloat)(1, 0), (cfloat)(0,0), (cfloat)(0,0), 
// 	(cfloat)(0, 0), (cfloat)(0, 0), (cfloat)(0,0), (cfloat)(1,0), 
// 	(cfloat)(0, 0), (cfloat)(0, 0), (cfloat)(1,0), (cfloat)(0,0)
// }; // 1
// constant cfloat X[] = { 
// 	(cfloat)(0, 0), (cfloat)(1, 0), 
// 	(cfloat)(1, 0), (cfloat)(0, 0) 
// }; // 2
// constant cfloat Y[] = { 
// 	(cfloat)(0, 0), (cfloat)(0, 0), 
// 	(cfloat)(0, 0), (cfloat)(0, 0) 
// }; // 3
// constant cfloat Z[] = { 
// 	(cfloat)(0, 0), (cfloat)(0, 0), 
// 	(cfloat)(0, 0), (cfloat)(0, 0) 
// }; // 4

constant int gateSizes[] = {
	4, 16, 4, 4 ,4
};
// // /****** END GATES ******/

// static const cfloat* gateForCode(int code) {
// 	if(code == 0) return &H;
// 	else if(code == 1) return &CNOT;
// 	else if(code == 2) return &X;
// 	else if(code == 3) return &Y;
// 	else if(code == 4) return &Z;
// 	return 0;
// }

// as per Kelly (2018)
static int nthCleared(int n, int t) {
  int mask = (1 << t) - 1;
  int notMask = ~mask;

  return (n & mask) | ((n & notMask) << 1);
}

static int nnthCleared(int n, int tCount, int* ts) {
  int mask = 1;
  for(int i = 0; i < tCount; i++)
    mask = mask << ts[i];

  mask -= 1;
  int notMask = ~mask;

  return (n & mask) | ((n & notMask) << tCount);
}

int nthInSequence(int n, int tCount, int* ts, int s) {
  if(s >= pow(2,tCount)) {
		/// TODO: throw an error or so
	}
  
  int f = nnthCleared(n, tCount, ts);

  for(int i = 0; i < tCount; i++) {
    if(s & (1<<i)) {
      f |= 1 << ts[i];
    }
  }
  
  return f;
}

/*** PROBLEM DEFINITION ***/
#define n 1 // number of qubits
#define N 2 // 2^n
#define G 1 // number of gates

#define SINGLE_QUBIT_GATE_LOOP_COUNT 1 // 2^(n-1)

#define K_SINGLE_MULTIPLIER_EACH_CALLS 1
#define K_DOUBLE_MULTIPLIER_EACH_CALLS 0


#define PROBLEM_SIZE 3
constant int problem[PROBLEM_SIZE] = {0, 0, 0};

channel cfloat outCh;

channel int singleMultiplier0GateCodeCh;
channel cvec2 singleMultiplier0InCh;
channel cvec2 singleMultiplier0OutCh;


__kernel void singleMultiplier0() {
	// printf("single multiplier 0 starting\n");

	for(int calls = 0; calls < K_SINGLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		// printf("single multiplier 0 listening for code..\n");
		int code = read_channel_altera(singleMultiplier0GateCodeCh);

		// /****** GATES ******/
		cfloat H[] = { 
			(cfloat)(M_SQRT1_2_F, 0), (cfloat)(M_SQRT1_2_F, 0), 
			(cfloat)(M_SQRT1_2_F, 0), (cfloat)(-M_SQRT1_2_F, 0) 
		}; // 0
		cfloat X[] = { 
			(cfloat)(0, 0), (cfloat)(1, 0), 
			(cfloat)(1, 0), (cfloat)(0, 0) 
		}; // 2
		cfloat Y[] = { 
			(cfloat)(0, 0), (cfloat)(0, 0), 
			(cfloat)(0, 0), (cfloat)(0, 0) 
		}; // 3
		cfloat Z[] = { 
			(cfloat)(0, 0), (cfloat)(0, 0), 
			(cfloat)(0, 0), (cfloat)(0, 0) 
		}; // 4

		cfloat *mat;

		if(code == 0) mat = H;
		else if(code == 2) mat = X;
		else if(code == 3) mat = Y;
		else if(code == 4) mat = Z;

		// cfloat vec[2];

		// #pragma unroll
		// for(int i = 0; i < 2; i++) {
		// 	// printf("single multiplier 0 listening for input...\n");
		// 	vec[i] = read_channel_altera(singleMultiplierInCh[0][i]);
		// }

		cvec2 inVec = read_channel_altera(singleMultiplier0InCh);

		cvec2 outVec;

		outVec.a = cadd(cmult(mat[0],inVec.a), cmult(mat[1],inVec.b));
		outVec.b = cadd(cmult(mat[2],inVec.a), cmult(mat[3],inVec.b));

		write_channel_altera(singleMultiplier0OutCh, outVec);

		// #pragma unroll
		// for(int i = 0; i < 2; i++) {
		// 	// printf("single multiplier 0 writing output...\n");
		// 	write_channel_altera(singleMultiplierOutCh[0][i], outVec[i]);
		// }

		// printf("single multiplier 0 ending\n");
	}
}


__kernel void mainKernel() {
	cfloat state[N] = {(cfloat)(1,0),(cfloat)(0,0)};

	// read the problem like a tape
	int tape = 0;
	for(int g = 0; g < G; g++) {
		int gateCode = problem[tape++];
		int gateSize = gateSizes[gateCode];

		if(gateSize == 4) {
			int qID = problem[tape++];
			int numberOfControls = problem[tape++];
			int c = numberOfControls == 1 ? problem[tape++] : 0;

			{ // SINGLE QUBIT GATE HANDLER 0
	int zero_state = nthCleared(0, qID);
	int one_state = zero_state | (1 << qID);

	cvec2 inVec;
	inVec.a = state[zero_state];
	inVec.b = state[one_state];

	int controlZero = 1; // true
	int controlOne = 1; // true

	if(numberOfControls == 1) {
		controlZero = (((1 << c) & zero_state) > 0) ? 1 : 0;
		controlOne = (((1 << c) & one_state) > 0) ? 1 : 0;
	}

	// activate the i-th multiplication kernel
	write_channel_altera(singleMultiplier0GateCodeCh, gateCode);
	write_channel_altera(singleMultiplier0InCh, inVec);
	
	// read out
	cvec2 outVec = read_channel_altera(singleMultiplier0OutCh);

	if(controlZero) state[zero_state] = outVec.a;
	if(controlOne) state[one_state] = outVec.b;
}



			// #pragma unroll
			// for(int i = 0; i < SINGLE_QUBIT_GATE_LOOP_COUNT; i++) {
			// 	int zero_state = nthCleared(i, qID);
			// 	int one_state = zero_state | (1 << qID);

			// 	cvec2 inVec;
			// 	inVec.a = state[zero_state];
			// 	inVec.b = state[one_state];

			// 	int controlZero = 1; // true
			// 	int controlOne = 1; // true

			// 	if(numberOfControls == 1) {
			// 		controlZero = (((1 << c) & zero_state) > 0) ? 1 : 0;
			// 		controlOne = (((1 << c) & one_state) > 0) ? 1 : 0;
			// 	}

			// 	// activate the i-th multiplication kernel
			// 	write_channel_altera(singleMultiplierGateCodeCh[i], gateCode);
			// 	write_channel_altera(singleMultiplierInCh[i], inVec);
				
			// 	// read out
			// 	cvec2 outVec = read_channel_altera(singleMultiplierOutCh[i]);

			// 	if(controlZero) state[zero_state] = outVec.a;
			// 	if(controlOne) state[one_state] = outVec.b;
			// }
		} 
		// else if(gateSize == 16) {
		// 	int qID1 = problem[tape++];
		// 	int qID2 = problem[tape++];
		// 	/// TODO: investigate why we need to flip these qubit IDs
		// 	int qIDs[2] = { qID2, qID1 };

		// 	for(int i = 0; i < pow(2,n-2); i++) {
		// 		// activate the i-th multiplication kernel
		// 		write_channel_altera(doubleMultiplierGateCodeCh[i], gateCode);

		// 		int stateIndices[4];
		// 		for(int j = 0; j < 4; j++) {
		// 			stateIndices[j] = nthInSequence(i, 2, qIDs, j);
		// 			write_channel_altera(doubleMultiplierInCh[i][j], state[stateIndices[j]]);
		// 		}

		// 		//read out
		// 		for(int j = 0; j < 4; j++) {
		// 			state[stateIndices[j]] = read_channel_altera(doubleMultiplierOutCh[i][j]);
		// 		}		
		// 	}
		// } 
		else {
			/// TODO: error: unsupported gate size
		}
	}

	// write final state to output
	#pragma unroll
	for(int i = 0; i < N; i++) {
		write_channel_altera(outCh, state[i]);
	}
}

__kernel void output(__global  float * restrict state_real, __global float * restrict state_imag) {
	for(int i = 0; i < N; i++) {
		cfloat amp = read_channel_altera(outCh);
		state_real[i] = creal(amp);
		state_imag[i] = cimag(amp);
	}
}
