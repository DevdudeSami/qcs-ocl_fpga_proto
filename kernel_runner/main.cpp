#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(X) assert(CL_SUCCESS == (X))

void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data);
unsigned char *load_file(const char* filename,size_t*size_ret);
void init(float *a, int size);

typedef struct {
	char *name;
	int numArgsIn;
	int numArgsOut;
	int *argsLengths;
	float **args;
	int *inIndices;
	int *outIndices;
} kernel;

kernel *createKernel(char *name, int numArgsIn, int numArgsOut, int *argsLengths, float **args, int *inIndices, int *outIndices) {
	kernel *k = (kernel *)malloc(sizeof(kernel));
	k->name = name;
	k->numArgsIn = numArgsIn;
	k->numArgsOut = numArgsOut;
	k->argsLengths = (int *)malloc(sizeof(int)*(numArgsIn+numArgsOut));
	memcpy(k->argsLengths, argsLengths, sizeof(int)*(numArgsIn+numArgsOut));
	k->args = args;
	k->inIndices = inIndices;
	k->outIndices = outIndices;
	return k;
}

void destroyKernel(kernel *k) {
	free(k->name);
	free(k->argsLengths);
	free(k->args);
	free(k->inIndices);
	free(k->outIndices);
	free(k);
}

void runKernel(kernel *k) {
	// Init OpenCL
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_program program;

	cl_kernel kernel;
	cl_command_queue commandQueue;
	
	cl_int status = 0;
  int num_errs = 0;
  cl_uint num_platforms = 0;

	//platform,device, context, command queue
	//---------------------------------------
	CHECK(clGetPlatformIDs(1, &platform, NULL));       
	//get platform vendor
	char cl_platform_vendor[1001];
	char cl_platform_version[51];
	CHECK (clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor, NULL));
	CHECK (clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 50, (void *)cl_platform_version, NULL));
	printf("HST::CL_PLATFORM_VENDOR:\t%s\t:: Version: %s\n",cl_platform_vendor, cl_platform_version);   

	printf("HST::Getting AOCL FPGA target device\n");
	CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,0) );

	context = clCreateContext(0, 1, &device, notify_print, 0, &status);
	CHECK(status);

	commandQueue = clCreateCommandQueue(context, device, 0, &status);
	CHECK(status);

	// no pipes needed for AOCL FPGA compilation

	// load kernel
	const unsigned char *binary = 0;
	cl_int bin_status = 0;
	size_t binary_len = 0;
	char aocx_name[strlen(k->name)+5]; 
	sprintf(aocx_name, "%s%s", k->name, ".aocx");
	printf("HST::Loading kernel binary %s ...\n", aocx_name);

	binary = load_file(aocx_name, &binary_len); 

	if ((binary == 0) || (binary_len == 0)) { 
		printf("HST::Error: unable to read %s into memory or the file was not found!\n", aocx_name);
		exit(-1);
	}

	program = clCreateProgramWithBinary(context,1,&device,&binary_len,&binary,&bin_status,&status);
	CHECK(status);

	// none needed for FPGA compilation
	const char *preProcFlags="";

	// create cl_kernel
	cl_kernel clKernel = clCreateKernel(program, k->name, &status);
	CHECK(status);

	cl_mem *buffers = (cl_mem *)malloc(sizeof(cl_mem)*(k->numArgsIn+k->numArgsOut));
	for(int i = 0; i < k->numArgsIn+k->numArgsOut; i++) {
		int len = (k->argsLengths)[i];
		buffers[i] = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len,0,&status); 
		CHECK(status);
	}

	printf("HST::Preparing kernels\n");

	for(int i = 0; i < k->numArgsIn; i++) {
		int argIndex = k->inIndices[i];
		float *arg = k->args[argIndex];
		int argLength = k->argsLengths[argLength];
		CHECK(clEnqueueWriteBuffer(commandQueue, buffers[argIndex], 0, 0, argLength*sizeof(float), arg, 0, 0, 0));
	}
	CHECK(clFinish(commandQueue));

	for(int i = 0; i < k->numArgsIn+k->numArgsOut; i++) {
		CHECK(clSetKernelArg(clKernel, i, sizeof(cl_mem), k->args[i]));
	}

	size_t dims[3] = {0, 0, 0};    
	//Currently I am working with Single work-instance kernels, so that
	//looping in the kernel can be explored, which is more suitable for FPGA targets
	dims[0] = 1;

	printf("HST::Enqueueing kernel with global size %d\n",(int)dims[0]);  
	CHECK(clEnqueueNDRangeKernel(commandQueue,clKernel,1,0,dims,0,0,0,0));

	CHECK(clFinish(commandQueue));
	
	//Read results
  //---------------------
  printf("HST::Reading results to host buffers...\n");
	for(int i = 0; i < k->numArgsOut; i++) {
		int argIndex = k->outIndices[i];
		float *arg = k->args[argIndex];
		int argLength = k->argsLengths[argIndex];
		CHECK(clEnqueueReadBuffer(commandQueue, buffers[argIndex], 1,0, argLength*sizeof(float), arg, 0,0,0));
	}

	// release buffers
	for(int i = 0; i < k->numArgsIn+k->numArgsOut; i++) {
		clReleaseMemObject(buffers[i]);
	}

	clReleaseProgram(program);
	clReleaseContext(context);
}

int main(int argc, char** argv) {
	// run the vector adder
	float *a;
	float *b;
	float *c;
	init(a, 10);
	init(b, 10);

	char *name = "vector_add";
	int argsLengths[3] = {10, 10, 10};
	float *args[3] = {a, b, c};
	int inIndices[2] = {0,1};
	int outIndices[1] = {2};
	kernel *k = createKernel(name, 2, 1, argsLengths, args, inIndices, outIndices);
	runKernel(k);
	destroyKernel(k);

	return 0;
}



////////////////////////////////////////
// HELPER FUNCTIONS
////////////////////////////////////////



float randFloat(float a, float b) {
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}

void init(float *a, int size) {      
  for(int i=0; i<size; i++) a[i] = randFloat(0, 10);
}

//-------------------------------------------------
//load_file
//-------------------------------------------------
unsigned char *load_file(const char* filename,size_t*size_ret)
{
   FILE* fp;
   int len;
   const size_t CHUNK_SIZE = 1000000;
   unsigned char *result;
   size_t r = 0;
   size_t w = 0;
   fp = fopen(filename,"rb");
   if ( !fp ) return 0;
   // Obtain file size.
   fseek(fp, 0, SEEK_END);
   len = ftell(fp);
   // Go to the beginning.
   fseek(fp, 0, SEEK_SET);
   // Allocate memory for the file data.
   result = (unsigned char*)malloc(len+CHUNK_SIZE);
   if ( !result )
   {
     fclose(fp);
     return 0;
   }
   // Read file.
   while ( 0 < (r=fread(result+w,1,CHUNK_SIZE,fp) ) )
   {
     w+=r;
   }
   fclose(fp);
   *size_ret = w;
   return result;
}


//-------------------------------------------------
// notify_print - needed for clCreateContext
//-------------------------------------------------

void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data)
{
   private_info = private_info;
   cb = cb;
   user_data = user_data;
   printf("HST::Error: %s\n", errinfo);
}



