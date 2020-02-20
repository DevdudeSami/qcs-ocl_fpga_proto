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
// constant cfloat H[] = { (0.707107, 0), (0.707107, 0), (0.707107, 0), (-0.707107, 0) }; // 0
// /****** END GATES ******/

// static cfloat[4] gateForCode(int code) {
// 	if(code == 0) return H;
// 	return nullptr;
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
#define n 2 // number of qubits
#define N 4 // 2^n
#define G 2 // number of gates
#define SINGLE_CALLS 1
#define DOUBLE_CALLS 1
constant int qTargetsCount[] = { 1, 2 }; // Number of target qubits of each gate
constant int qTargets[] = { 0, 0, 1 }; // The qubit ID targets of each gate
constant int gateSizes[] = { 4, 16 };
// #define totalTargetCounts 2 // sum of qTargetsCount
#define totalGatesLength 20 // sum of gateSizes

#define MULTIPLIER_KERNELS 2

#define SINGLE_QUBIT_GATE_SIZE 4
#define DOUBLE_QUBIT_GATE_SIZE 16

channel cfloat inCh;
channel cfloat outCh;

channel int singleQubitMetaCh;
channel cfloat singleQubitGateInCh;
channel cfloat singleQubitGateOutCh;

channel int doubleQubitMetaCh;
channel cfloat doubleQubitGateInCh;
channel cfloat doubleQubitGateOutCh;

channel int twoMultiplierOpCodeCh[MULTIPLIER_KERNELS];
channel cfloat twoMultiplierInCh[MULTIPLIER_KERNELS];
channel cfloat twoMultiplierOutCh[MULTIPLIER_KERNELS];

#define TWO_MULTIPLIER_EACH_CALLS 1
__kernel void twoMultiplier() {

	size_t gid = get_global_id(0);
	printf("TWO KERNEL... %i\n", gid);

	for(int calls = 0; calls < TWO_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		int opCode = read_channel_altera(twoMultiplierOpCodeCh[gid]);
		printf("TWO KERNEL: opCode = %i ... \n", opCode);

		if(opCode == 0) break; // exit if opCode is 0

		// otherwise start run

		cfloat mat[4];
		cfloat vec[2];

		for(int i = 0; i < 4; i++) {
			mat[i] = read_channel_altera(twoMultiplierInCh[gid]);
		}
		for(int i = 0; i < 2; i++) {
			vec[i] = read_channel_altera(twoMultiplierInCh[gid]);
		}

		cfloat outVec[2];

		outVec[0] = cadd(cmult(mat[0],vec[0]), cmult(mat[1],vec[1]));
		outVec[1] = cadd(cmult(mat[2],vec[0]), cmult(mat[3],vec[1]));

		for(int i = 0; i < 2; i++) {
			write_channel_altera(twoMultiplierOutCh[gid], outVec[i]);
		}
	}
}


// M x M gate
// N-dimensional state
__kernel void kernelInput(__global  float * restrict gates_r, __global  float * restrict gates_i, __global  float * restrict state_r, __global  float * restrict state_i) {

	printf("INPUT KERNEL...\n");

	// write the gate to the channel
	for(int i = 0; i < totalGatesLength; i++) {
		write_channel_altera(inCh, (cfloat)(gates_r[i], gates_i[i]));
	}

	// write the state to the channel
	for(int i = 0; i < N; i++) {
		write_channel_altera(inCh, (cfloat)(state_r[i], state_i[i]));
	}
}

__kernel void singleQubitGate() {
	printf("SINGLE QUBIT GATE KERNEL...\n");

	for(int calls = 0; calls < SINGLE_CALLS; calls++) {
		// read in the target qubit ID
		int qID = (int)read_channel_altera(singleQubitMetaCh);

		// read in the gate
		cfloat gate[SINGLE_QUBIT_GATE_SIZE];
		for(int i = 0; i < SINGLE_QUBIT_GATE_SIZE; i++) {
			gate[i] = read_channel_altera(singleQubitGateInCh);
		}

		// read in state vector
		cfloat state[N];
		for(int i = 0; i < N; i++) state[i] = read_channel_altera(singleQubitGateInCh);

		// apply the computation
		for(int i = 0; i < pow(2,n-1); i++) {
			int zero_state = nthCleared(i, qID);
			int one_state = zero_state | (1 << qID);
			printf("qID %i\t zero_state %i\t one_state %i\n", qID, zero_state, one_state);

			printf("SINGLE QUBIT GATE KERNEL: i = %i...\n", i);
			cfloat zero_amp = state[zero_state];
			cfloat one_amp = state[one_state];

			// activate the i-th multiplication kernel
			write_channel_altera(twoMultiplierOpCodeCh[i], 1);
			for(int j = 0; j < SINGLE_QUBIT_GATE_SIZE; j++) {
				write_channel_altera(twoMultiplierInCh[i], gate[j]);
			}
			write_channel_altera(twoMultiplierInCh[i], zero_amp);
			write_channel_altera(twoMultiplierInCh[i], one_amp);
			
			// read out
			state[zero_state] = read_channel_altera(twoMultiplierOutCh[i]);
			state[one_state] = read_channel_altera(twoMultiplierOutCh[i]);
		}

		// write out output
		for(int i = 0; i < N; i++) {
			write_channel_altera(singleQubitGateOutCh, state[i]);
		}
	}
}

__kernel void doubleQubitGate() {
	int calls = 0;
	while(calls < DOUBLE_CALLS) {
		// read in the target qubit IDs
		int qID1 = (int)read_channel_altera(doubleQubitMetaCh);
		int qID2 = (int)read_channel_altera(doubleQubitMetaCh);
		int qIDs[2] = { qID2, qID1 };

		// read in the gate
		cfloat gate[DOUBLE_QUBIT_GATE_SIZE];
		for(int i = 0; i < DOUBLE_QUBIT_GATE_SIZE; i++) {
			gate[i] = read_channel_altera(doubleQubitGateInCh);
		}

		// read in state vector
		cfloat state[N];
		for(int i = 0; i < N; i++) {
			state[i] = read_channel_altera(doubleQubitGateInCh);
		}


		// apply the computation
		for(int i = 0; i < pow(2,n-2); i++) {
			int stateIndices[4];
			cfloat substate[4];
			for(int j = 0; j < 4; j++) {
				stateIndices[j] = nthInSequence(i, 2, qIDs, j);
				substate[j] = state[stateIndices[j]];
			}
		printf("qIDs... %i\t%i\n", qIDs[0], qIDs[1]);
		printf("stateIndices... %i\t%i\t%i\t%i\n", stateIndices[0], stateIndices[1], stateIndices[2], stateIndices[3]);
		printf("substate... %f\t%f\t%f\t%f\n", creal(substate[0]), creal(substate[1]), creal(substate[2]), creal(substate[3]));

			for(int j = 0; j < 4; j++) {
				cfloat sum = (0,0);
				for(int k = 0; k < 4; k++) {
					sum = cadd(sum, cmult(gate[j*4+k], substate[k]));
				}
				state[stateIndices[j]] = sum;
			}
		}

		// write out output
		for(int i = 0; i < N; i++) {
			write_channel_altera(doubleQubitGateOutCh, state[i]);
		}

		calls++;
	}
}


__kernel void kernelCompute() {
	// read in all gates
	cfloat gates[totalGatesLength];
	for(int i = 0; i < totalGatesLength; i++) gates[i] = read_channel_altera(inCh);

	// read in the input state
	cfloat state[N];
	for(int i = 0; i < N; i++) state[i] = read_channel_altera(inCh);

	// loop through the gates
	int consumedGates = 0;
	int consumedTargets = 0;
	for(int g = 0; g < G; g++) {
		int targetCount = qTargetsCount[g];
		if(targetCount == 1) {
			// activate single qubit kernel by writing the target qubit ID 
			write_channel_altera(singleQubitMetaCh, qTargets[consumedTargets]);
			// write the gate and the state to the kernel
			for(int i = 0; i < SINGLE_QUBIT_GATE_SIZE; i++) {
				write_channel_altera(singleQubitGateInCh, gates[consumedGates+i]);
			}
			for(int i = 0; i < N; i++) {
				write_channel_altera(singleQubitGateInCh, state[i]);
			}

			// now read on the output channel of the kernel
			for(int i = 0; i < N; i++) {
				state[i] = read_channel_altera(singleQubitGateOutCh);
			}

			consumedGates += SINGLE_QUBIT_GATE_SIZE;
		} else if(targetCount == 2) {
			// activate double qubit kernel by writing the target qubit IDs
			write_channel_altera(doubleQubitMetaCh, qTargets[consumedTargets]);
			write_channel_altera(doubleQubitMetaCh, qTargets[consumedTargets+1]);
			// write the gate and the state to the kernel
			for(int i = 0; i < DOUBLE_QUBIT_GATE_SIZE; i++) {
				write_channel_altera(doubleQubitGateInCh, gates[consumedGates+i]);
			}
			for(int i = 0; i < N; i++) {
				write_channel_altera(doubleQubitGateInCh, state[i]);
			}

			// now read on the output channel of the kernel
			for(int i = 0; i < N; i++) {
				state[i] = read_channel_altera(doubleQubitGateOutCh);
			}

			consumedGates += DOUBLE_QUBIT_GATE_SIZE;
		} 
		else {
			/// TODO: error
		}
		consumedTargets += targetCount;
	}

	// write final state to output
	for(int i = 0; i < N; i++) {
		write_channel_altera(outCh, state[i]);
	}
}

// __kernel void kernelCompute() {
// 	cfloat gates[totalGatesLength];
// 	for(int i = 0; i < totalGatesLength; i++) gates[i] = (cfloat)(read_channel_altera(realCh), read_channel_altera(imagCh));

// 	cfloat state[N];
// 	for(int i = 0; i < N; i++) state[i] = (cfloat)(read_channel_altera(realCh), read_channel_altera(imagCh));


// 	// consume qTargets as a tape
// 	int qIDTapeHead = 0;
// 	int gateTapeHead = 0;
// 	for(int g = 0; g < G; g++) {
// 		int targetCount = qTargetsCount[g];
// 		int *qIDs = (int *)malloc(sizeof(int)*targetCount);
// 		for(int i = 0; i < targetCount; i++) {
// 			qIDs[i] = qTargets[qIDTapeHead++];
// 			printf("\tqID %d...\n", qIDs[i]);
// 		}

// 		int targetN = pow(2, targetCount);
// 		cfloat *gate = (cfloat *)malloc(sizeof(cfloat)*targetN*targetN);
// 		for(int i = 0; i < targetN*targetN; i++) {
// 			gate[i] = gates[gateTapeHead++];
// 		}

// 		// 2^(n-targetCount) needed for the loop
// 		int loopLim = pow(2, n - targetCount);

// 		// multiply
// 		for(int i = 0; i < loopLim; i++) {
// 			printf("\tProcessing loop index %d...\n", i);
// 			int *stateIndices = (int *)malloc(sizeof(int)*targetN);
// 			cfloat *amps = (cfloat *)malloc(sizeof(cfloat)*targetN);

// 			for(int j = 0; j < targetN; j++) {
// 				int stateIndex = nthInSequence(i, targetCount, qIDs, N - 1 - j);
// 				printf("\t\tstateIndex %d...\n", stateIndex);
// 				stateIndices[j] = stateIndex;
// 				amps[j] = state[stateIndex];
// 				printf("\t\tamps[%d] %f...\n", j, creal(amps[j]));
// 			}

// 			for(int i = 0; i < targetN; i++) {
// 				cfloat sum = (cfloat)(0,0);
// 				for(int j = 0; j < targetN; j++) {
// 					printf("\t\t\tgate... %f\n", creal(gate[targetN*i+j]));
// 					printf("\t\t\tamp... %f\n", creal(amps[j]));
// 					sum = cadd(sum, cmult(gate[targetN*i+j],amps[j]));
// 				}
// 				state[stateIndices[i]] = sum;
// 			}
// 		}
// 	}

// 	// write the new state to the output channel
// 	for(int i = 0; i < N; i++) {
// 		write_channel_altera(realOutCh, creal(state[i]));
// 		write_channel_altera(imagOutCh, cimag(state[i]));
// 	}
// }

__kernel void kernelOutput(__global  float * restrict state_real, __global float * restrict state_imag) {
	for(int i = 0; i < N; i++) {
		cfloat amp = read_channel_altera(outCh);
		state_real[i] = creal(amp);
		state_imag[i] = cimag(amp);
	}
}



// static read_cfloat_array_channel(int size, cfloat *arr, channel float realCh, channel float imagCh) {
// 	for(int i = 0; i < size; i++) {
// 		arr[i] = (cfloat)(read_channel_altera(realCh), read_channel_altera(imagCh));
// 	}
// }

// static write_cfloat_array_channel(int size, cfloat* arr, channel float realCh, channel float imagCh) {
// 	for(int i = 0; i < size; i++) {
// 		write_channel_altera(realCh, creal(arr[i]));
// 		write_channel_altera(imagCh, cimag(arr[i]));
// 	}
// }