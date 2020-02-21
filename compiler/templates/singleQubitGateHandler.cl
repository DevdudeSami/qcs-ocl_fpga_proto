{ // SINGLE QUBIT GATE HANDLER {{i}}
	int zero_state = nthCleared({{i}}, qID);
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
	write_channel_altera(singleMultiplier{{i}}GateCodeCh, gateCode);
	write_channel_altera(singleMultiplier{{i}}InCh, inVec);
	
	// read out
	cvec2 outVec = read_channel_altera(singleMultiplier{{i}}OutCh);

	if(controlZero) state[zero_state] = outVec.a;
	if(controlOne) state[one_state] = outVec.b;
}

