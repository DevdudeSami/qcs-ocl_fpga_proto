//-----------------
//Problem Size
//-----------------
//Make sure it matches size in main.cpp
#define SIZE      10

__kernel void vector_add(__global float *restrict x, 
                         __global float *restrict y, 
                         __global float *restrict z)
{
    // int i = get_global_id(0);

    // add the vector elements
    for(int i=0; i<SIZE; i++) {
        z[i] = x[i] + y[i];
    }
}

