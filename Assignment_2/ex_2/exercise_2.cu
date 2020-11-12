#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define ARRAY_SIZE 5000000
#define TPB 256


void fill_data(float *var)
{
	int i;

	if(var == NULL)
		return;

	for(i=0; i<ARRAY_SIZE; i++)
	{
		
		var[i] = 100 * (float)((float)rand()/RAND_MAX); 
	}		
}

void saxpy_cpu(float *x, float *y, float A)
{
	struct timeval start_time;
	struct timeval stop_time;
	int cnt = 0;

	gettimeofday(&start_time, NULL);
	for(cnt=0; cnt<ARRAY_SIZE; cnt++)
	{
		y[cnt] = (A* x[cnt]) + y[cnt];
	}
	gettimeofday(&stop_time, NULL);

	printf("Total time of Execution in CPU: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
}

__global__ void kernel(float *x, float *y, float a, int size)
{	
	int i = (blockIdx.x*blockDim.x)+threadIdx.x;
	if(i < size)	
		y[i] = (a*x[i]) + y[i];
	__syncthreads();
}

void saxpy_gpu(float *x, float *y, float A)
{
	struct timeval start_time;
        struct timeval stop_time;

        gettimeofday(&start_time, NULL);

        kernel<<<(ARRAY_SIZE/TPB)+1, TPB>>>(x, y, A, ARRAY_SIZE);
	cudaDeviceSynchronize();      
	gettimeofday(&stop_time, NULL);

	printf("Total time of Execution in GPU: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
}


int main()
{
	float *X = NULL;
	float *Y = NULL;
	float *Y_GPU = NULL;
	float A = 2.3;

	float *gpuX = NULL;
	float *gpuY = NULL;

	X = (float*)malloc(ARRAY_SIZE*sizeof(float));
	Y = (float*)malloc(ARRAY_SIZE*sizeof(float));
	Y_GPU = (float*)malloc(ARRAY_SIZE*sizeof(float));

	fill_data(X);
	fill_data(Y);

	cudaMalloc(&gpuX, ARRAY_SIZE*sizeof(float));
        cudaMalloc(&gpuY, ARRAY_SIZE*sizeof(float));

        cudaMemcpy(gpuX, X, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);      
       	cudaMemcpy(gpuY, Y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	saxpy_cpu(X, Y, A);
	printf("Computing SAXPY on the CPU... Done!\n\n");		

	saxpy_gpu(gpuX, gpuY, A);
	printf("Computing SAXPY on the GPU... Done!\n\n");	

	cudaMemcpy(Y_GPU, gpuY, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);	
		
	int i = 0;

	for(i=0; i<ARRAY_SIZE; i++)
	{
		if(((Y[i] - Y_GPU[i]) < -0.05) || ((Y[i] - Y_GPU[i]) > 0.05))
		{
			printf("Comparing the output of each implementation.. Mismatch at index %d\n",i);
				break;
		}
	}
	if(i == ARRAY_SIZE)
		printf("Comparing the output of each implementation.. Correct\n");

	cudaFree(gpuX);
	cudaFree(gpuY);

	free(X);
	free(Y);
	return 0;
}
