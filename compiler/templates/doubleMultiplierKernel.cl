__kernel void doubleMultiplier{{n}}() {
	for(int calls = 0; calls < K_DOUBLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		int code = read_channel_altera(doubleMultiplierGateCodeCh[{{n}}]);

		cfloat *mat = gateForCode(code);

		cfloat vec[4];

		for(int i = 0; i < 4; i++) {
			vec[i] = read_channel_altera(doubleMultiplierInCh[{{n}}][i]);
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
			write_channel_altera(doubleMultiplierOutCh[{{n}}][i], outVec[i]);
		}
	}
}