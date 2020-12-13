// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

unsigned int NUM_PARTICLES;
unsigned int NUM_ITERATIONS;
unsigned int BLOCK_SIZE;

typedef struct {
	float posX;
	float posY;
	float posZ;
}position;

typedef struct {
	float velX;
	float velY;
	float velZ;
}velocity;

typedef struct {
	position pos;
	velocity vel;
}Particle;

const char *mykernel =
"typedef struct {\n"
"        float posX;\n"
"        float posY;\n"
"        float posZ;\n"
"}position;\n"
"typedef struct {\n"
"        float velX;\n"
"        float velY;\n"
"        float velZ;\n"
"}velocity;\n"
"typedef struct {\n"
"        position pos;\n"
"        velocity vel;\n"
"}Particle;\n"
"\n"
"__kernel void updatePar(__global Particle *p,\n"
"	__global int *NPar,\n"
"	__global int *NIter)\n"
"{ int i = get_global_id(0);\n"
"int j = 0;\n"
"if(i < (*NPar))\n"
"{\n"
"for(j=0; j<(*NIter); j++)\n"
"{\n"
"	(i+j)%2 ? (p[i].vel.velX += (5*(i+j))%100) : (p[i].vel.velX -= (5*(i+j))%100);\n"
"	(i+j)%2 ? (p[i].vel.velY += (3*(i+j))%100) : (p[i].vel.velY -= (3*(i+j))%100);\n"
"	(i+j)%2 ? (p[i].vel.velZ += (7*(i+j))%100) : (p[i].vel.velZ -= (7*(i+j))%100);\n"
"	p[i].pos.posX = p[i].pos.posX + p[i].vel.velX;\n"
"	p[i].pos.posY = p[i].pos.posY + p[i].vel.velY;\n"
"	p[i].pos.posZ = p[i].pos.posZ + p[i].vel.velZ;\n"
"}}}\n";

/*Only a one-time step to fill data to the array of structure(i.e. Particle)*/
void fill_data(Particle *p1, Particle *p2)
{
	int i;
	for(i=0; i< NUM_PARTICLES; i++)
	{
		p1[i].pos.posX = 10*((float)rand()/RAND_MAX);	
		p1[i].pos.posY = 10*((float)rand()/RAND_MAX);		
		p1[i].pos.posZ = 10*((float)rand()/RAND_MAX);
		
		p1[i].vel.velX = 100*((float)rand()/RAND_MAX);		
		p1[i].vel.velY = 100*((float)rand()/RAND_MAX);		
		p1[i].vel.velZ = 100*((float)rand()/RAND_MAX);

		p2[i].pos.posX = p1[i].pos.posX;
		p2[i].pos.posY = p1[i].pos.posY;
		p2[i].pos.posZ = p1[i].pos.posZ;

		p2[i].vel.velX = p1[i].vel.velX;
		p2[i].vel.velY = p1[i].vel.velY;
		p2[i].vel.velZ = p1[i].vel.velZ;		
	}
}

/*update velocity w.r.t to common pattern () and then 
 * update position w.r.t formula given: p.x = p.x + v.x.dt where dt=1 */
void update_velocity_position_in_CPU(Particle *p)
{
	int i,j;
	struct timeval start_time;
	struct timeval stop_time;

	gettimeofday(&start_time, NULL);
	for(j=0; j<NUM_ITERATIONS; j++)
	{
		for(i=0; i<NUM_PARTICLES; i++)
		{	
			(i+j)%2 ? (p[i].vel.velX += (5*(i+j))%100) : (p[i].vel.velX -= (5*(i+j))%100);
			(i+j)%2 ? (p[i].vel.velY += (3*(i+j))%100) : (p[i].vel.velY -= (3*(i+j))%100);
			(i+j)%2 ? (p[i].vel.velZ += (7*(i+j))%100) : (p[i].vel.velZ -= (7*(i+j))%100);

			p[i].pos.posX = p[i].pos.posX + p[i].vel.velX;
			p[i].pos.posY = p[i].pos.posY + p[i].vel.velY;
			p[i].pos.posZ = p[i].pos.posZ + p[i].vel.velZ;	
		}
	}
	gettimeofday(&stop_time, NULL);

	printf("Total time of Execution in CPU: %ld usec\n\n", 
		(stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
}

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		printf("Please pass 3 arguments\n");
		printf("1st - NUM_PARTICLES\n");
		printf("2nd - NUM_ITERATIONS\n");
		printf("3rd - BLOCK_SIZE");
		exit(1);
	}
	NUM_PARTICLES = atoi(argv[1]);
	NUM_ITERATIONS = atoi(argv[2]);
	BLOCK_SIZE = atoi(argv[3]);
	
	/* Variables */
        Particle *par = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
	Particle *par_cpu = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
	int i = 0;

	cl_platform_id*	platforms;
	cl_uint		n_platform;  

	// Find OpenCL Platforms
	cl_int err = clGetPlatformIDs(0, NULL, &n_platform);
	CHK_ERROR(err);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*n_platform);
	err = clGetPlatformIDs(n_platform, platforms, NULL);
	CHK_ERROR(err);

	// Find and sort devices
	cl_device_id *device_list;
	cl_uint n_devices;
	err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);
	CHK_ERROR(err);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*n_devices);
	err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
	CHK_ERROR(err);

	// Create and initialize an OpenCL context 
	cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);
	CHK_ERROR(err);

	// Create a command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
	CHK_ERROR(err);   

	/********************************* My code here ***********************************/

	fill_data(par, par_cpu);
	
	update_velocity_position_in_CPU(par_cpu);	

	cl_mem par_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_PARTICLES*sizeof(Particle), NULL, &err);
	cl_mem nPar_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	cl_mem nIter_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&mykernel, NULL, &err);

	err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "Build error: %s\n", buffer);
		return 0;
	}

	cl_kernel kernel = clCreateKernel(program, "updatePar", &err);

	size_t n_workitem;
	if((NUM_PARTICLES % BLOCK_SIZE) != 0)
		n_workitem = NUM_PARTICLES + BLOCK_SIZE - (NUM_PARTICLES % BLOCK_SIZE); 
	else
        	n_workitem = NUM_PARTICLES;
        size_t workgroup_size = BLOCK_SIZE;

	struct timeval start_time;
        struct timeval stop_time;

	gettimeofday(&start_time, NULL);

	//Transfer Data to Device
	err = clEnqueueWriteBuffer(cmd_queue, par_dev, CL_TRUE, 0, NUM_PARTICLES*sizeof(Particle), par, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(cmd_queue, nPar_dev, CL_TRUE, 0, sizeof(int), &NUM_PARTICLES, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(cmd_queue, nIter_dev, CL_TRUE, 0, sizeof(int), &NUM_ITERATIONS, 0, NULL, NULL);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&par_dev);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&nPar_dev);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&nIter_dev);

        //Launch Kernel
        err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL);

	//Transfer the data to Y_gpu
	err = clEnqueueReadBuffer(cmd_queue, par_dev, CL_TRUE, 0, NUM_PARTICLES*sizeof(Particle), par, 0, NULL, NULL);

	gettimeofday(&stop_time, NULL);

	err = clFlush(cmd_queue);
	err = clFinish(cmd_queue);

	printf("Total time of Execution in GPU: %ld usec\n\n",
		(stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));

        for(i=0; i<NUM_PARTICLES; i++)
        {
                if(((par_cpu[i].pos.posX - par[i].pos.posX) < -0.05) || ((par_cpu[i].pos.posX - par[i].pos.posX) > 0.05))
                {
			if(((par_cpu[i].pos.posY - par[i].pos.posY) < -0.05) || ((par_cpu[i].pos.posY - par[i].pos.posY) > 0.05))
			{
				if(((par_cpu[i].pos.posZ - par[i].pos.posZ) < -0.05) || ((par_cpu[i].pos.posZ - par[i].pos.posZ) > 0.05))
				{
                        		printf("Comparing the output of each implementation.. Mismatch at index %d\n",i);
                                	break;
				}
			}
                }
        }
        if(i == NUM_PARTICLES)
                printf("Comparing the output of each implementation.. Correct\n");

		
	/**********************************************************************************/


	// Finally, release all that we have allocated.
	err = clReleaseCommandQueue(cmd_queue);
	CHK_ERROR(err);
	err = clReleaseContext(context);
	CHK_ERROR(err);
	free(platforms);
	free(device_list);

	return 0;
}


// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes

const char* clGetErrorString(int errorCode)
{
	switch (errorCode)
	{
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case -69: return "CL_INVALID_PIPE_SIZE";
		case -70: return "CL_INVALID_DEVICE_QUEUE";
		case -71: return "CL_INVALID_SPEC_ID";
		case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
		case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
		case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
		case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
		case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
		case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
		case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
		case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
		case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
		case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
		case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
		case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
		case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
		case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
		case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
		case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
		case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
		case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
		default: return "CL_UNKNOWN_ERROR";
		}
}
