__kernel void singleMultiplier{{n}}() {
	printf("single multiplier {{n}} starting\n");

	for(int calls = 0; calls < K_SINGLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		printf("single multiplier {{n}} listening for code..\n");
		int code = read_channel_altera(singleMultiplierGateCodeCh[{{n}}]);

		cfloat *mat = gateForCode(code);

		cfloat vec[2];

		for(int i = 0; i < 2; i++) {
			printf("single multiplier {{n}} listening for input...\n");
			vec[i] = read_channel_altera(singleMultiplierInCh[{{n}}][i]);
		}

		cfloat outVec[2];

		outVec[0] = cadd(cmult(mat[0],vec[0]), cmult(mat[1],vec[1]));
		outVec[1] = cadd(cmult(mat[2],vec[0]), cmult(mat[3],vec[1]));

		for(int i = 0; i < 2; i++) {
			printf("single multiplier {{n}} writing output...\n");
			write_channel_altera(singleMultiplierOutCh[{{n}}][i], outVec[i]);
		}

		printf("single multiplier {{n}} ending\n");
	}
}
