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
#define VECTOR_ADD   0
#define NAME_K0     "vector_add"
const char* kernel_names[NKERNELS] = {
	NAME_K0
};

//multiple kernels (possibly over  multiple devices) required multiple command queues
cl_kernel kernels[NKERNELS];
cl_command_queue commands[NKERNELS]; 

#define SIZE 10

void init(float a[SIZE]);
void writeResults(float a[SIZE], float b[SIZE], float c[SIZE]);
unsigned char *load_file(const char* filename,size_t*size_ret);
void notify_print(const char* errinfo, const void* private_info, size_t cb, void *user_data);

int main(int argc, char** argv) {
	float a[SIZE];
	float b[SIZE];
	float c[SIZE];

	init(a);
	init(b);

	//-------------------------------------------------
	//opencl run
	//-------------------------------------------------

  //ocl device memory pointers
  //---------------------
  cl_mem aDev = 0;
  cl_mem bDev = 0;
  cl_mem cDev = 0;
  
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
	const char *         aocx_name = "vector_add.aocx";
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
  aDev = clCreateBuffer(context,CL_MEM_READ_WRITE,SIZE*sizeof(float),0,&status); CHECK(status);
  bDev = clCreateBuffer(context,CL_MEM_READ_WRITE,SIZE*sizeof(float),0,&status); CHECK(status);
	cDev = clCreateBuffer(context,CL_MEM_READ_WRITE,SIZE*sizeof(float),0,&status); CHECK(status);

  // Prepare Kernel, Args
  //---------------------
  printf("HST::Preparing kernels\n");   
  int cl_wi=0;

	// write initial data to buffer on device (0th command queue for 0th kernel)
  CHECK(clEnqueueWriteBuffer(commands[0],aDev,0,0,SIZE*sizeof(float),a,0,0,0));
  CHECK(clEnqueueWriteBuffer(commands[0],bDev,0,0,SIZE*sizeof(float),b,0,0,0));
  CHECK(clFinish(commands[0]));

	size_t dims[3] = {0, 0, 0};    
	//Currently I am working with Single work-instance kernels, so that
	//looping in the kernel can be explored, which is more suitable for FPGA targets
	dims[0] = 1;
	
	CHECK(clSetKernelArg(kernels[VECTOR_ADD],0,sizeof(cl_mem),&aDev));
	CHECK(clSetKernelArg(kernels[VECTOR_ADD],1,sizeof(cl_mem),&bDev));
	CHECK(clSetKernelArg(kernels[VECTOR_ADD],2,sizeof(cl_mem),&cDev));

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
  CHECK(clEnqueueReadBuffer(commands[VECTOR_ADD],cDev,1,0,SIZE*sizeof(float),c,0,0,0));

	printf("HST::Checking results...\n");   
  writeResults(a, b, c);

	//Post-processing
  //---------------------
  clReleaseMemObject(aDev);
  clReleaseMemObject(bDev);
  clReleaseMemObject(cDev);
  //clReleaseKernel(kernel);
  clReleaseProgram(my_program);
  clReleaseContext(context);

	return 0;
}

void init(float a[SIZE]) {      
  for(int i=0; i<SIZE; i++) a[i] = rand();
}

void writeResults(float a[SIZE], float b[SIZE], float c[SIZE]) {
  FILE *fp = fopen("out.dat", "w");
  FILE *fperror = fopen("error.log", "w");
	
	// linear display with indices, and also compare results
	fprintf(fp, "-------------------------------------------------------------\n");
	fprintf(fp, "       i ::  a,   b , c\n");
	fprintf(fp, "-------------------------------------------------------------\n");

	for(int i=0; i<SIZE; i++)
		fprintf (fp, "\t%5d\t::\t%10f\t%10f\t%10f\n", i,  a[i],  b[i], c[i]);
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








