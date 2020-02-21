__kernel void singleMultiplier{{n}}() {
	// printf("single multiplier {{n}} starting\n");

	for(int calls = 0; calls < K_SINGLE_MULTIPLIER_EACH_CALLS; calls++) {
		// listen for an op code
		// printf("single multiplier {{n}} listening for code..\n");
		int code = read_channel_altera(singleMultiplier{{n}}GateCodeCh);

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
		// 	// printf("single multiplier {{n}} listening for input...\n");
		// 	vec[i] = read_channel_altera(singleMultiplierInCh[{{n}}][i]);
		// }

		cvec2 inVec = read_channel_altera(singleMultiplier{{n}}InCh);

		cvec2 outVec;

		outVec.a = cadd(cmult(mat[0],inVec.a), cmult(mat[1],inVec.b));
		outVec.b = cadd(cmult(mat[2],inVec.a), cmult(mat[3],inVec.b));

		write_channel_altera(singleMultiplier{{n}}OutCh, outVec);

		// #pragma unroll
		// for(int i = 0; i < 2; i++) {
		// 	// printf("single multiplier {{n}} writing output...\n");
		// 	write_channel_altera(singleMultiplierOutCh[{{n}}][i], outVec[i]);
		// }

		// printf("single multiplier {{n}} ending\n");
	}
}
