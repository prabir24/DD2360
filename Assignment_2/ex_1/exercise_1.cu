#include<stdio.h>
#define TBP 256

__global__ void hello_world()
{
	printf("Hello World! My threadId is %d\n",threadIdx.x);
	__syncthreads();
}

int main()
{
	hello_world<<<1,TBP>>>();
	cudaDeviceSynchronize();
	return 0;
}

