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

//-------------------------------------------------
//globals
//-------------------------------------------------
const unsigned char *binary = 0;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_program my_program;

#define NKERNELS    1
#define M_V_MUL   0
#define NAME_K0     "matrix_vector_mul"
const char* kernel_names[NKERNELS] = {
	NAME_K0
};

//multiple kernels (possibly over  multiple devices) required multiple command queues
cl_kernel kernels[NKERNELS];
cl_command_queue commands[NKERNELS]; 

const int M = 10;
const int N = 10;

void init(float *a, int size);
void writeResults(float mat[M*N], float vecIn[M], float vecOut[N]);
unsigned char *load_file(const char* filename,size_t*size_ret);
void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data);

int main(int argc, char** argv) {
	float mat[M*N];
	float vecIn[M];
	float vecOut[N];

	init(mat, M*N);
	init(vecIn, M);

	//-------------------------------------------------
	//opencl run
	//-------------------------------------------------

  //ocl device memory pointers
  //---------------------
  cl_mem matDev = 0;
  cl_mem vecInDev = 0;
  cl_mem vecOutDev = 0;
  
  //other variables
  //---------------------
  cl_int status = 0;
  int num_errs = 0;
  int i;
  
  cl_uint num_platforms = 0;
  
  //platform,device, context, command queue
  //---------------------------------------
  CHECK(clGetPlatformIDs(1, &platform, NULL));       
  //get platform vendor
  char cl_platform_vendor[1001];
  char cl_platform_version[51];
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor, NULL) );
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VERSION, 50, (void *)cl_platform_version, NULL) );
  printf("HST::CL_PLATFORM_VENDOR:\t%s\t:: Version: %s\n",cl_platform_vendor, cl_platform_version);   

  printf("HST::Getting AOCL FPGA target device\n");
  CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,0) );

  context = clCreateContext(0, 1, &device, notify_print, 0, &status);
  CHECK(status);

	//Create separate queue for each kernel, even if on same device  
	for(i=0; i<NKERNELS; i++) {
		commands[i] = clCreateCommandQueue(context, device, 0, &status);
		CHECK(status);
	}

  //pipes
  //---------------------------------------
	//AOCL creates channels (pipes)  in the kernel file global scope

  // load kernel
  //------------
	// For FPGA target, the kernels are compiled offline (via aoc compiler), and binary is loaded   
	cl_int               bin_status = 0;
	const unsigned char* my_binary;
	size_t               my_binary_len = 0;
	const char *         aocx_name = "matrix_vector_mul.aocx";
	printf("HST::Loading kernel binary %s ...\n", aocx_name);
	
	my_binary = load_file(aocx_name, &my_binary_len); 

	if ((my_binary == 0) || (my_binary_len == 0))
	{ 
		printf("HST::Error: unable to read %s into memory or the file was not found!\n", aocx_name);
		exit(-1);
	}

	my_program = clCreateProgramWithBinary(context,1,&device,&my_binary_len,&my_binary,&bin_status,&status);
	CHECK(status);

	//for CPU/GPU options, we are compiling kernel at runtime, so
  //we need to pass it pre-processor flags here
	const char *preProcFlags="";

  printf("HST::Building program\n");
  CHECK(clBuildProgram(my_program,1,&device,preProcFlags,0,0));

	printf("HST::Creating kernel(s)\n");
  //create kernel(s)
  for (i=0; i<NKERNELS; i++) {  
		kernels[i] = clCreateKernel(my_program, kernel_names[i], &status);
		CHECK(status);
  }    

  // cl buffers
  //---------------------
  printf("HST::Creating cl (device) buffers\n");
  matDev = clCreateBuffer(context,CL_MEM_READ_WRITE,M*N*sizeof(float),0,&status); CHECK(status);
  vecInDev = clCreateBuffer(context,CL_MEM_READ_WRITE,M*sizeof(float),0,&status); CHECK(status);
	vecOutDev = clCreateBuffer(context,CL_MEM_READ_WRITE,N*sizeof(float),0,&status); CHECK(status);

  // Prepare Kernel, Args
  //---------------------
  printf("HST::Preparing kernels\n");   

	// write initial data to buffer on device (0th command queue for 0th kernel)
  CHECK(clEnqueueWriteBuffer(commands[0],matDev,0,0,M*N*sizeof(float),mat,0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],vecInDev,0,0,M*sizeof(float),vecIn,0,0,0));
  CHECK(clFinish(commands[0]));

	size_t dims[3] = {0, 0, 0};    
	//Currently I am working with Single work-instance kernels, so that
	//looping in the kernel can be explored, which is more suitable for FPGA targets
	dims[0] = 1;
	
	CHECK(clSetKernelArg(kernels[M_V_MUL],0,sizeof(int),&M));
	CHECK(clSetKernelArg(kernels[M_V_MUL],1,sizeof(int),&N));
	CHECK(clSetKernelArg(kernels[M_V_MUL],2,sizeof(cl_mem),&matDev));
	CHECK(clSetKernelArg(kernels[M_V_MUL],3,sizeof(cl_mem),&vecInDev));
	CHECK(clSetKernelArg(kernels[M_V_MUL],4,sizeof(cl_mem),&vecOutDev));

	//Launch Kernel
  //-------------
  printf("HST::Enqueueing kernel with global size %d\n",(int)dims[0]);  
  for (int i=0; i<NKERNELS; i++) {
    CHECK(clEnqueueNDRangeKernel(commands[i],kernels[i],1,0,dims,0,0,0,0));
  }
  
	for (int i=0; i<NKERNELS; i++)
		CHECK(clFinish(commands[i]));

  //Read results
  //---------------------
  printf("HST::Reading results to host buffers...\n");
  CHECK(clEnqueueReadBuffer(commands[M_V_MUL],vecOutDev,1,0,N*sizeof(float),vecOut,0,0,0));

	printf("HST::Writing results...\n");   
  writeResults(mat, vecIn, vecOut);

	//Post-processing
  //---------------------
  clReleaseMemObject(matDev);
  clReleaseMemObject(vecInDev);
  clReleaseMemObject(vecOutDev);
  //clReleaseKernel(kernel);
  clReleaseProgram(my_program);
  clReleaseContext(context);

	return 0;
}

void init(float *a, int size) {      
  for(int i=0; i<size; i++) a[i] = rand();
}

void writeResults(float mat[M*N], float vecIn[M], float vecOut[N]) {
  FILE *fp = fopen("out.dat", "w");
  FILE *fperror = fopen("error.log", "w");
	
	// linear display with indices, and also compare results
	fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     mat:\n");
	fprintf(fp, "-------------------------------------------------------------\n");

  for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			fprintf(fp, "\t%5f", mat[j+M*i]);
		}
    fprintf(fp, "\n");
	}

  fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     vecIn:\n");
	fprintf(fp, "-------------------------------------------------------------\n");
	for(int i=0; i<M; i++)
		fprintf (fp, "\t%5f\n", vecIn[i]);

  fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "     vecOut:\n");
	fprintf(fp, "-------------------------------------------------------------\n");
	for(int i=0; i<N; i++)
		fprintf (fp, "\t%5f\n", vecOut[i]);
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








