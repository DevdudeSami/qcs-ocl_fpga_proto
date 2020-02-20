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

// /****** GATES ******/
constant cfloat H[] = { 
	(cfloat)(M_SQRT1_2_F, 0), (cfloat)(M_SQRT1_2_F, 0), 
	(cfloat)(M_SQRT1_2_F, 0), (cfloat)(-M_SQRT1_2_F, 0) 
}; // 0
constant cfloat CNOT[] = { 
	(cfloat)(1, 0), (cfloat)(0, 0), (cfloat)(0,0), (cfloat)(0,0),
	(cfloat)(0, 0), (cfloat)(1, 0), (cfloat)(0,0), (cfloat)(0,0), 
	(cfloat)(0, 0), (cfloat)(0, 0), (cfloat)(0,0), (cfloat)(1,0), 
	(cfloat)(0, 0), (cfloat)(0, 0), (cfloat)(1,0), (cfloat)(0,0)
}; // 1
constant cfloat X[] = { 
	(cfloat)(0, 0), (cfloat)(1, 0), 
	(cfloat)(1, 0), (cfloat)(0, 0) 
}; // 2
constant cfloat Y[] = { 
	(cfloat)(0, 0), (cfloat)(0, 0), 
	(cfloat)(0, 0), (cfloat)(0, 0) 
}; // 3
constant cfloat Z[] = { 
	(cfloat)(0, 0), (cfloat)(0, 0), 
	(cfloat)(0, 0), (cfloat)(0, 0) 
}; // 4

constant int gateSizes[] = {
	4, 16, 4, 4 ,4
};
// /****** END GATES ******/

static const cfloat* gateForCode(int code) {
	if(code == 0) return &H;
	else if(code == 1) return &CNOT;
	else if(code == 2) return &X;
	else if(code == 3) return &Y;
	else if(code == 4) return &Z;
	return 0;
}

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
#define n 2 // number of qubits
#define N 4 // 2^n
#define G 2 // number of gates

#define K_SINGLE_MULTIPLIER_EACH_CALLS 1
#define K_DOUBLE_MULTIPLIER_EACH_CALLS 1


#define PROBLEM_SIZE 5
constant int problem[PROBLEM_SIZE] = {0, 0, 1, 0, 1};

channel cfloat outCh;

channel int singleMultiplierGateCodeCh[2];
channel cfloat singleMultiplierInCh[2][2];
channel cfloat singleMultiplierOutCh[2][2];
channel int doubleMultiplierGateCodeCh[1];
channel cfloat doubleMultiplierInCh[1][4];
channel cfloat doubleMultiplierOutCh[1][4];


__kernel void singleMultiplier0() {
	printf("single multiplier 0 starting\n");

	for(int calls = 0; calls < K_SINGLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		printf("single multiplier 0 listening for code..\n");
		int code = read_channel_altera(singleMultiplierGateCodeCh[0]);

		cfloat *mat = gateForCode(code);

		cfloat vec[2];

		for(int i = 0; i < 2; i++) {
			printf("single multiplier 0 listening for input...\n");
			vec[i] = read_channel_altera(singleMultiplierInCh[0][i]);
		}

		cfloat outVec[2];

		outVec[0] = cadd(cmult(mat[0],vec[0]), cmult(mat[1],vec[1]));
		outVec[1] = cadd(cmult(mat[2],vec[0]), cmult(mat[3],vec[1]));

		for(int i = 0; i < 2; i++) {
			printf("single multiplier 0 writing output...\n");
			write_channel_altera(singleMultiplierOutCh[0][i], outVec[i]);
		}

		printf("single multiplier 0 ending\n");
	}
}
__kernel void singleMultiplier1() {
	printf("single multiplier 1 starting\n");

	for(int calls = 0; calls < K_SINGLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		printf("single multiplier 1 listening for code..\n");
		int code = read_channel_altera(singleMultiplierGateCodeCh[1]);

		cfloat *mat = gateForCode(code);

		cfloat vec[2];

		for(int i = 0; i < 2; i++) {
			printf("single multiplier 1 listening for input...\n");
			vec[i] = read_channel_altera(singleMultiplierInCh[1][i]);
		}

		cfloat outVec[2];

		outVec[0] = cadd(cmult(mat[0],vec[0]), cmult(mat[1],vec[1]));
		outVec[1] = cadd(cmult(mat[2],vec[0]), cmult(mat[3],vec[1]));

		for(int i = 0; i < 2; i++) {
			printf("single multiplier 1 writing output...\n");
			write_channel_altera(singleMultiplierOutCh[1][i], outVec[i]);
		}

		printf("single multiplier 1 ending\n");
	}
}
__kernel void doubleMultiplier0() {
	for(int calls = 0; calls < K_DOUBLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		int code = read_channel_altera(doubleMultiplierGateCodeCh[0]);

		cfloat *mat = gateForCode(code);

		cfloat vec[4];

		for(int i = 0; i < 4; i++) {
			vec[i] = read_channel_altera(doubleMultiplierInCh[0][i]);
		}

		cfloat outVec[4];

		for(int j = 0; j < 4; j++) {
			cfloat sum = (cfloat)(0,0);
			for(int k = 0; k < 4; k++) {
				sum = cadd(sum, cmult(mat[j*4+k], vec[k]));
			}
			outVec[j] = sum;
		}

		for(int i = 0; i < 4; i++) {
			write_channel_altera(doubleMultiplierOutCh[0][i], outVec[i]);
		}
	}
}

__kernel void mainKernel() {
	cfloat state[N] = {(cfloat)(1,0),(cfloat)(0,0),(cfloat)(0,0),(cfloat)(0,0)};

	// read the problem like a tape
	int tape = 0;
	for(int g = 0; g < G; g++) {
		int gateCode = problem[tape++];
		int gateSize = gateSizes[gateCode];

		if(gateSize == 4) {
			int qID = problem[tape++];

			for(int i = 0; i < pow(2,n-1); i++) {
				int zero_state = nthCleared(i, qID);
				int one_state = zero_state | (1 << qID);
				cfloat zero_amp = state[zero_state];
				cfloat one_amp = state[one_state];

				// activate the i-th multiplication kernel
				write_channel_altera(singleMultiplierGateCodeCh[i], gateCode);
				write_channel_altera(singleMultiplierInCh[i][0], zero_amp);
				write_channel_altera(singleMultiplierInCh[i][1], one_amp);
				
				// read out
				state[zero_state] = read_channel_altera(singleMultiplierOutCh[i][0]);
				state[one_state] = read_channel_altera(singleMultiplierOutCh[i][1]);
			}
		} 
		else if(gateSize == 16) {
			int qID1 = problem[tape++];
			int qID2 = problem[tape++];
			/// TODO: investigate why we need to flip these qubit IDs
			int qIDs[2] = { qID2, qID1 };

			for(int i = 0; i < pow(2,n-2); i++) {
				// activate the i-th multiplication kernel
				write_channel_altera(doubleMultiplierGateCodeCh[i], gateCode);

				int stateIndices[4];
				for(int j = 0; j < 4; j++) {
					stateIndices[j] = nthInSequence(i, 2, qIDs, j);
					write_channel_altera(doubleMultiplierInCh[i][j], state[stateIndices[j]]);
				}

				//read out
				for(int j = 0; j < 4; j++) {
					state[stateIndices[j]] = read_channel_altera(doubleMultiplierOutCh[i][j]);
				}		
			}
		} 
		else {
			/// TODO: error: unsupported gate size
		}
	}

	// write final state to output
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
