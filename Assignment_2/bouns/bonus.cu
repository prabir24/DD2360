#include<stdio.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#include<sys/time.h>
#include<math.h>

unsigned int NUM_ITER = 1000000000;

unsigned int NUM_ITERATIONS = 1000;
unsigned int BLOCK_SIZE = 192;
unsigned int GRID_SIZE = (NUM_ITER/(NUM_ITERATIONS*BLOCK_SIZE));


__global__ void gpu_random(curandState *states, int Niterations, int *count) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int seed = id;
	double x,y,z;
	count[id] = 0;
        curand_init(seed, id, 0, &states[id]);
	
	for(int i=0; i<Niterations; i++)
	{
		x = curand_uniform(&states[id]);
		y = curand_uniform(&states[id]);
		z = sqrt((x*x) + (y*y));

		if(z <= 1.0)
			count[id] += 1;
	}	
        __syncthreads();
}


int main(int argc, char *argv[])
{
	if(argc != 4)
	{
		printf("No. of arguments to be passed should be 2,1st arg as total Num of Iteration, 2nd as num of iteartion per thread, 3rd as BLock Size\n");
		exit(1);
	}

	NUM_ITER = atoi(argv[1]);
	NUM_ITERATIONS = atoi(argv[2]);
	BLOCK_SIZE = atoi(argv[3]);

	struct timeval start_time;
        struct timeval stop_time;

	curandState *dev_random;
	cudaMalloc((void**)&dev_random, BLOCK_SIZE*GRID_SIZE*sizeof(curandState));

	int *countCPU = NULL;
	countCPU = (int*)malloc(BLOCK_SIZE*GRID_SIZE*sizeof(int));

	int *countGPU = NULL;
	cudaMalloc(&countGPU, BLOCK_SIZE*GRID_SIZE*sizeof(int));

	gettimeofday(&start_time, NULL);			
	gpu_random<<<GRID_SIZE, BLOCK_SIZE>>>(dev_random, NUM_ITERATIONS, countGPU);
        cudaDeviceSynchronize();

	cudaMemcpy(countCPU, countGPU, BLOCK_SIZE*GRID_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

	int finalCount;
	for(int i=0; i<(BLOCK_SIZE*GRID_SIZE); i++)
		finalCount += countCPU[i];

	double pi;
	pi = ((double)finalCount / (double)NUM_ITER) * 4.0;
	gettimeofday(&stop_time, NULL);

	printf("The result of PI is %lf\n",pi);	

	printf("Total time of Execution to calculate PI using GPU is: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));

	cudaFree(dev_random);
	cudaFree(countGPU);
	free(countCPU);
	return 0;
}

