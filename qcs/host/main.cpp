#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex>
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

const int n = 2;
const int N = pow(2, n);
const int M = 2; 

void writeResults(cxf mat[M*M], cxf vecIn[N], cxf vecOut[N]);
unsigned char *load_file(const char* filename,size_t*size_ret);
void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data);

int main(int argc, char** argv)  {

	// initialize output files
  FILE *fp, *fperror;
  fp      = fopen("out.dat", "w");
  fperror = fopen("error.log", "w");

	cxf gate[M*M] = { 0, 1, 1, 0 };
	cxf stateIn[N] = { 0, 0, 1, 0 };
	cxf stateOut[N];
	int qID = 0;

	// define device memory pointers
	cl_mem gateDev = 0;
	cl_mem stateInDev = 0;
	cl_mem stateOutDev = 0;

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
  gateDev  = clCreateBuffer(context,CL_MEM_READ_WRITE,M*M*sizeof(cxf),0,&status); CHECK(status);
  stateInDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);
  stateOutDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(cxf),0,&status); CHECK(status);

	// write initial data to buffer on device (0th command queue for 0th kernel)
	printf("HST::Preparing kernels\n");   
  CHECK(clEnqueueWriteBuffer(commands[0],gateDev,0,0,M*M*sizeof(cxf),gate,0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],stateInDev,0,0,N*sizeof(cxf),stateIn,0,0,0));
  CHECK(clFinish(commands[0]));

	// set input kernel args
	CHECK(clSetKernelArg(kernels[K_INPUT], 0, sizeof(int), &M));
	CHECK(clSetKernelArg(kernels[K_INPUT], 1, sizeof(int), &N));
	CHECK(clSetKernelArg(kernels[K_INPUT], 2, sizeof(cl_mem), &gateDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 3, sizeof(cl_mem), &stateInDev));
	CHECK(clSetKernelArg(kernels[K_INPUT], 4, sizeof(int), &qID));
  
	// set output kernel args
	CHECK(clSetKernelArg(kernels[K_OUTPUT], 0, sizeof(int), &N));
	CHECK(clSetKernelArg(kernels[K_OUTPUT], 1, sizeof(cl_mem), &stateOutDev));

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
  CHECK(clEnqueueReadBuffer(commands[K_OUTPUT],stateOutDev,1,0,N*sizeof(cxf),stateOut,0,0,0));

	printf("HST::Writing results...\n");   
  writeResults(gate, stateIn, stateOut);

	// release memory
	clReleaseMemObject(gateDev);
	clReleaseMemObject(stateInDev);
	clReleaseMemObject(stateOutDev);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}

void writeResults(cxf mat[M*M], cxf vecIn[N], cxf vecOut[N]) {
  FILE *fp = fopen("out.dat", "w");
  FILE *fperror = fopen("error.log", "w");
	
	// fprintf(fp, "-------------------------------------------------------------\n");
	// fprintf(fp, "     mat:\n");
	// fprintf(fp, "-------------------------------------------------------------\n");

  // for(int i = 0; i < M; i++) {
	// 	for(int j = 0; j < M; j++) {
	// 		fprintf(fp, "\t%f%+fi",mat[j+M*i].real(), mat[j+M*i].imag());
	// 	}
  //   fprintf(fp, "\n");
	// }

  // fprintf(fp, "-------------------------------------------------------------\n");
	// fprintf(fp, "     vecIn:\n");
	// fprintf(fp, "-------------------------------------------------------------\n");
	// for(int i=0; i<M; i++)
	// 	fprintf (fp, "\t%f%+fi\n", vecIn[i].real(), vecIn[i].imag());

  // fprintf(fp, "-------------------------------------------------------------\n");
	// fprintf(fp, "     vecOut:\n");
	// fprintf(fp, "-------------------------------------------------------------\n");
	// for(int i=0; i<M; i++)
	// 	fprintf (fp, "\t%f%+fi\n", vecOut[i].real(), vecOut[i].imag());
	
	
	fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     mat:\n");
	fprintf(fp, "-------------------------------------------------------------\n");

  for(int i = 0; i < M; i++) {
		for(int j = 0; j < M; j++) {
			fprintf(fp, "\t%f",mat[j+M*i]);
		}
    fprintf(fp, "\n");
	}

  fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     vecIn:\n");
	fprintf(fp, "-------------------------------------------------------------\n");
	for(int i=0; i<N; i++)
		fprintf (fp, "\t%f\n", vecIn[i]);

  fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     vecOut:\n");
	fprintf(fp, "-------------------------------------------------------------\n");
	for(int i=0; i<N; i++)
		fprintf (fp, "\t%f\n", vecOut[i]);
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

