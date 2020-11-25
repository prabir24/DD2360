#include<stdio.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#include<sys/time.h>

unsigned int NUM_PARTICLES = 100000;
unsigned int NUM_ITERATIONS = 10;
unsigned int BLOCK_SIZE = 192;
unsigned int GRID_SIZE = ((NUM_PARTICLES/BLOCK_SIZE) + 1);

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

__global__ void particle_kernel_per_iteration(Particle *p, int Nparticles)
{
        int i = (blockIdx.x*blockDim.x)+threadIdx.x;

        if(i < Nparticles) {
		p[i].pos.posId.x += p[i].vel.velId.x;
             	p[i].pos.posId.y += p[i].vel.velId.y;
               	p[i].pos.posId.z += p[i].vel.velId.z;
        }
        __syncthreads();
}


void update_velocity_position_in_gpu()
{
        struct timeval start_time;
        struct timeval stop_time;

	Particle *Par = NULL;

	cudaMallocManaged(&Par, NUM_PARTICLES*sizeof(Particle));

	fill_data(Par);

        //Start time
        gettimeofday(&start_time, NULL);

        for(int i=0; i<NUM_ITERATIONS; i++)
        {
		//Launch kernel		
		particle_kernel_per_iteration<<<GRID_SIZE, BLOCK_SIZE>>>(Par, NUM_PARTICLES);
		cudaDeviceSynchronize();

		//Update Velocity in Host before copying data to GPU Memory
		for(int j=0; j<NUM_PARTICLES;j++)
                {
                        Par[j].vel.velId.x = 100*((float)rand()/RAND_MAX);
                        Par[j].vel.velId.y = 100*((float)rand()/RAND_MAX);
                        Par[j].vel.velId.z = 100*((float)rand()/RAND_MAX);
                }
        }

	//Stop time
        gettimeofday(&stop_time, NULL);
	
        printf("Total time of Execution in GPU: %ld msec\n\n",
                ((stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec))/1000);
	cudaFree(Par);
}

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		printf("No. of arguments to be passed should be 2 i.e. 1st as NUM_PARTICLES and 2nd as BLOCK_SIZE\n");
		exit(1);
	}

	NUM_PARTICLES = atoi(argv[1]);
	BLOCK_SIZE = atoi(argv[2]);

	update_velocity_position_in_gpu();

	return 0;
}

