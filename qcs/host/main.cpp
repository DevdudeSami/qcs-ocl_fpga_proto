#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define CHECK(X) assert(CL_SUCCESS == (X))

#define NKERNELS    3
#define K_INPUT     0
#define K_COMPUTE   1
#define K_OUTPUT    2
#define NAME_K0 "kernelInput"
#define NAME_K1 "kernelCompute"
#define NAME_K2 "kernelOutput"
const char* kernel_names[NKERNELS] = {
	NAME_K0, 
	NAME_K1, 
	NAME_K2
};

//multiple kernels (possibly over  multiple devices) required multiple command queues
cl_kernel kernels[NKERNELS];
cl_command_queue commands[NKERNELS]; 

typedef float cxf;

//-------------------------------------------------
//globals
//-------------------------------------------------
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_program program;

const int n = 3;
const int N = pow(2, n);
const int G = 2; // number of gates

void writeResults(cxf vecOut_r[N], cxf vecOut_i[N]);
unsigned char *load_file(const char* filename,size_t*size_ret);
void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data);

int main(int argc, char** argv)  {

	// initialize output files
  FILE *fp, *fperror;
  fp      = fopen("out.dat", "w");
  fperror = fopen("error.log", "w");

	/// TODO: Rewrite this
	int qTargetsCount[G] = { 1, 1 };
	int totalTargetCounts = 0;
	int totalGatesLength = 0;
	for(int i = 0; i < G; i++) {
		totalTargetCounts += qTargetsCount[i];
		totalGatesLength += pow(pow(2,qTargetsCount[i]),2);
	}
	///////

	vector<int> qTargets { 0, 2 };

	vector<cxf> gates_r { 0, 1, 1, 0, 0, 1, 1, 0 }; // two X's
	vector<cxf> gates_i { 0, 0, 0, 0, 0, 0, 0, 0 };
	cxf stateIn_r[N] = { 1, 0, 0, 0 };
	cxf stateIn_i[N] = { 0, 0, 0, 0 };
	cxf stateOut_r[N];
	cxf stateOut_i[N];
	
	
	// define device memory pointers
	cl_mem qTargetsCountDev = 0;
	cl_mem qTargetsDev = 0;
	cl_mem gates_rDev = 0;
	cl_mem gates_iDev = 0;
	cl_mem stateIn_rDev = 0;
	cl_mem stateIn_iDev = 0;
	// output memory pointers
	cl_mem stateOut_rDev = 0;
	cl_mem stateOut_iDev = 0;

  cl_int status = 0;
  int num_errs = 0;


	//platform,device, context, command queue
  //---------------------------------------
  CHECK ( clGetPlatformIDs  (1, &platform, NULL) );       
  //get platform vendor
  char cl_platform_vendor[1001];
  char cl_platform_version[51];
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor, NULL) );
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VERSION, 50, (void *)cl_platform_version, NULL) );
  printf("HST::CL_PLATFORM_VENDOR:\t%s\t:: Version: %s\n",cl_platform_vendor, cl_platform_version);   

	printf("HST::Getting AOCL FPGA target device\n");
  CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,0) );

	context = clCreateContext( 0, 1, &device, notify_print, 0, &status );
  CHECK( status );

	// Create separate queue for each kernel, even if on same device  
	for(int i=0; i<NKERNELS; i++) {
			commands[i] = clCreateCommandQueue( context, device, 0, &status );
		CHECK( status );
	}

	// load kernels
	cl_int bin_status = 0;
  const unsigned char* binary;
  size_t binary_len = 0;
	const char * aocx_name = "kernels.aocx";
  printf("HST::Loading kernel binary %s ...\n", aocx_name);
  
  binary = load_file(aocx_name, &binary_len); 

  if ((binary == 0) || (binary_len == 0))
  { 
    printf("HST::Error: unable to read %s into memory or the file was not found!\n", aocx_name);
    exit(-1);
  }

  program = clCreateProgramWithBinary(context,1,&device,&binary_len,&binary,&bin_status,&status);
  CHECK(status);

	// build the program
	const char *preProcFlags="";

  printf("HST::Building program\n");
  CHECK( clBuildProgram(program,1,&device,preProcFlags,0,0) );

  //create kernel(s)
	printf("HST::Creating kernel(s)\n");
  for (int i=0; i<NKERNELS; i++) {  
   kernels[i] = clCreateKernel(program, kernel_names[i], &status);
   CHECK(status);
  }    

	// create device memory buffer
	printf("HST::Creating cl (device) buffers\n");
  qTargetsCountDev = clCreateBuffer(context,CL_MEM_READ_WRITE,G*sizeof(int),0,&status); CHECK(status);
  qTargetsDev = clCreateBuffer(context,CL_MEM_READ_WRITE,totalTargetCounts*sizeof(int),0,&status); CHECK(status);
  gates_rDev = clCreateBuffer(context,CL_MEM_READ_WRITE,totalGatesLength*sizeof(cxf),0,&status); CHECK(status);
  gates_iDev = clCreateBuffer(context,CL_MEM_READ_WRITE,totalGatesLength*sizeof(cxf),0,&status); CHECK(status);
  stateIn_rDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);
  stateIn_iDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);
  stateOut_rDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);
  stateOut_iDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);

	// write initial data to buffer on device (0th command queue for 0th kernel)
	printf("HST::Preparing kernels\n");   
  CHECK(clEnqueueWriteBuffer(commands[0],qTargetsCountDev,0,0,G*sizeof(int),qTargetsCount,0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],qTargetsDev,0,0,totalTargetCounts*sizeof(cxf),&qTargets[0],0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],gates_rDev,0,0,totalGatesLength*sizeof(cxf),&gates_r[0],0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],gates_iDev,0,0,totalGatesLength*sizeof(cxf),&gates_i[0],0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],stateIn_rDev,0,0,N*sizeof(cxf),stateIn_r,0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],stateIn_iDev,0,0,N*sizeof(cxf),stateIn_i,0,0,0));
  CHECK(clFinish(commands[0]));

	// set input kernel args
	CHECK(clSetKernelArg(kernels[K_INPUT], 0, sizeof(int), &G));
	CHECK(clSetKernelArg(kernels[K_INPUT], 1, sizeof(int), &n));
	CHECK(clSetKernelArg(kernels[K_INPUT], 2, sizeof(cl_mem), &qTargetsCountDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 3, sizeof(cl_mem), &qTargetsDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 4, sizeof(cl_mem), &gates_rDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 5, sizeof(cl_mem), &gates_iDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 6, sizeof(cl_mem), &stateIn_rDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 7, sizeof(cl_mem), &stateIn_iDev));
  
	// set output kernel args
	CHECK(clSetKernelArg(kernels[K_OUTPUT], 0, sizeof(int), &N));
	CHECK(clSetKernelArg(kernels[K_OUTPUT], 1, sizeof(cl_mem), &stateOut_rDev));
	CHECK(clSetKernelArg(kernels[K_OUTPUT], 2, sizeof(cl_mem), &stateOut_iDev));

	// single work-instance kernels
	size_t dims[3] = {0, 0, 0};    
  dims[0] = 1 ;

	//Launch Kernel
  //-------------
  printf("HST::Enqueueing kernel with global size %d\n",(int)dims[0]);  
  for (int i=0; i<NKERNELS; i++) {
    CHECK(clEnqueueNDRangeKernel(commands[i],kernels[i],1,0,dims,0,0,0,0));
  }

	// wait for kernels to finish
	for (int i=0; i<NKERNELS; i++)
		CHECK(clFinish(commands[i]));

	//Read results
  //---------------------
  printf("HST::Reading results to host buffers...\n");
  CHECK(clEnqueueReadBuffer(commands[K_OUTPUT],stateOut_rDev,1,0,N*sizeof(cxf),stateOut_r,0,0,0));
  CHECK(clEnqueueReadBuffer(commands[K_OUTPUT],stateOut_iDev,1,0,N*sizeof(cxf),stateOut_i,0,0,0));

	printf("HST::Writing results...\n");   
  writeResults(stateOut_r, stateOut_i);

	// release memory
	clReleaseMemObject(qTargetsCountDev);
	clReleaseMemObject(qTargetsDev);
	clReleaseMemObject(gates_rDev);
	clReleaseMemObject(gates_iDev);
	clReleaseMemObject(stateIn_rDev);
	clReleaseMemObject(stateIn_iDev);
	clReleaseMemObject(stateOut_rDev);
	clReleaseMemObject(stateOut_iDev);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}

void writeResults(cxf vecOut_r[N], cxf vecOut_i[N]) {
  FILE *fp = fopen("out.dat", "w");
  FILE *fperror = fopen("error.log", "w");

  fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     vecOut:\n");
	fprintf(fp, "-------------------------------------------------------------\n");
	for(int i=0; i<N; i++)
		fprintf (fp, "\t%f%+fi\n", vecOut_r[i], vecOut_i[i]);
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
   // Go to the beginning.ç
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

