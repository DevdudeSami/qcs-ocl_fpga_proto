//-----------------
//Problem Size
//-----------------
//Make sure it matches size in main.cpp

__kernel void vector_add(float *size, __global float *restrict x, __global float *restrict y, __global float *restrict z)
{
    // add the vector elements
    for(int i=0; i<(*size); i++) {
        z[i] = x[i] + y[i];
    }
}

