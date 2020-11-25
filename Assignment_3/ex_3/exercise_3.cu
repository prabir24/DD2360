#include<stdio.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#include<sys/time.h>

unsigned int NUM_PARTICLES = 1000000;
unsigned int NUM_ITERATIONS = 10;
unsigned int BLOCK_SIZE = 192;
//unsigned int GRID_SIZE = ((NUM_PARTICLES/BLOCK_SIZE) + 1);
unsigned int NUM_STREAMS = 10;


typedef struct {
	float3 posId;
}position;

typedef struct {
	float3 velId;
}velocity;

typedef struct {
	position pos;
	velocity vel;
}Particle;


void fill_data(Particle *p)
{
	for(int i=0; i< NUM_PARTICLES; i++)
	{
		p[i].pos.posId.x = 10*((float)rand()/RAND_MAX);	
		p[i].pos.posId.y = 10*((float)rand()/RAND_MAX);		
		p[i].pos.posId.z = 10*((float)rand()/RAND_MAX);
		
		p[i].vel.velId.x = 100*((float)rand()/RAND_MAX);		
		p[i].vel.velId.y = 100*((float)rand()/RAND_MAX);		
		p[i].vel.velId.z = 100*((float)rand()/RAND_MAX);		
	}
}

__global__ void particle_kernel_per_iteration(Particle *p, int offset, int streamSize)
{
        int i = (blockIdx.x*blockDim.x)+threadIdx.x;

        if(i < streamSize) {
		p[offset + i].pos.posId.x += p[offset + i].vel.velId.x;
             	p[offset + i].pos.posId.y += p[offset + i].vel.velId.y;
               	p[offset + i].pos.posId.z += p[offset + i].vel.velId.z;
        }
        __syncthreads();
}


void update_velocity_position_in_gpu(Particle *p)
{
        struct timeval start_time;
        struct timeval stop_time;

	Particle *gPar = NULL;

	cudaMalloc(&gPar, NUM_PARTICLES * sizeof(Particle));

	unsigned long streamSize = NUM_PARTICLES/NUM_STREAMS;
	unsigned long streamBytes = streamSize * sizeof(Particle);

	cudaStream_t stream[NUM_STREAMS];
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamCreate(&stream[i]);


        //Start time
        gettimeofday(&start_time, NULL);
#ifdef TYPE1
        for(int i=0; i<NUM_ITERATIONS; i++)
        {
		for(int s=0; s<NUM_STREAMS; s++)
		{
			unsigned long offset = s * streamSize;
			// Copy Data to GPU Memory Asynchronously
        		cudaMemcpyAsync(&gPar[offset], &p[offset], streamBytes, cudaMemcpyHostToDevice, stream[s]);

			//Launch kernel		
			particle_kernel_per_iteration<<<((streamSize/BLOCK_SIZE) + 1), BLOCK_SIZE, 0, stream[s]>>>(gPar, offset, streamSize);
			//cudaDeviceSynchronize();
		
			//Copy Data back to Host
			cudaMemcpyAsync(&p[offset], &gPar[offset], streamBytes, cudaMemcpyDeviceToHost, stream[s]);
		}
		cudaDeviceSynchronize();

		//Update Velocity in Host before copying data to GPU Memory
		for(int j=0; j<NUM_PARTICLES;j++)
                {
                        p[j].vel.velId.x = 100*((float)rand()/RAND_MAX);
                        p[j].vel.velId.y = 100*((float)rand()/RAND_MAX);
                        p[j].vel.velId.z = 100*((float)rand()/RAND_MAX);
                }
        }
#else
	for(int i=0; i<NUM_ITERATIONS; i++)
        {
                for(int s=0; s<NUM_STREAMS; s++)
                {
                        unsigned long offset = s * streamSize;
                        // Copy Data to GPU Memory Asynchronously
                        cudaMemcpyAsync(&gPar[offset], &p[offset], streamBytes, cudaMemcpyHostToDevice, stream[s]);
		}

		for(int s=0; s<NUM_STREAMS; s++)
                {
                        unsigned long offset = s * streamSize;
                        //Launch kernel         
                        particle_kernel_per_iteration<<<((streamSize/BLOCK_SIZE) + 1), BLOCK_SIZE, 0, stream[s]>>>(gPar, offset, streamSize);
		}

		for(int s=0; s<NUM_STREAMS; s++)
                {
                        unsigned long offset = s * streamSize;
                        //Copy Data back to Host
                        cudaMemcpyAsync(&p[offset], &gPar[offset], streamBytes, cudaMemcpyDeviceToHost, stream[s]);
                }
                cudaDeviceSynchronize();

                //Update Velocity in Host before copying data to GPU Memory
                for(int j=0; j<NUM_PARTICLES;j++)
                {
                        p[j].vel.velId.x = 100*((float)rand()/RAND_MAX);
                        p[j].vel.velId.y = 100*((float)rand()/RAND_MAX);
                        p[j].vel.velId.z = 100*((float)rand()/RAND_MAX);
                }
        }
#endif

	//Stop time
	gettimeofday(&stop_time, NULL);

        for(int i=0; i<NUM_STREAMS; i++)
                cudaStreamDestroy(stream[i]);
	
	cudaFree(gPar);

        printf("Total time of Execution in GPU: %ld msec\n\n",
                ((stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec))/1000);	
}

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		printf("No. of arguments to be passed should be 2 i.e. 1st as NUM_PARTICLES and 2nd as NUM_STREAMS\n");
		exit(1);
	}

	NUM_PARTICLES = atoi(argv[1]);
	NUM_STREAMS = atoi(argv[2]);
	
	Particle *par = NULL;
	cudaMallocHost(&par, NUM_PARTICLES*sizeof(Particle));
	
	fill_data(par);

	update_velocity_position_in_gpu(par);

	cudaFree(par);

	return 0;
}

