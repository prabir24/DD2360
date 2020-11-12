#include<stdio.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#include<sys/time.h>

unsigned int NUM_PARTICLES = 1000;
unsigned int NUM_ITERATIONS = 100;
unsigned int BLOCK_SIZE = 512;
unsigned int GRID_SIZE = ((NUM_PARTICLES)/BLOCK_SIZE);

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


/*Only a one-time step to fill data to the array of structure(i.e. Particle)*/
void fill_data(Particle *p)
{
	for(int i=0; i< NUM_PARTICLES; i++)
	{
		p->pos.posX = 10*((float)rand()/RAND_MAX);	
		p->pos.posY = 10*((float)rand()/RAND_MAX);		
		p->pos.posZ = 10*((float)rand()/RAND_MAX);
		
		p->vel.velX = 100*((float)rand()/RAND_MAX);		
		p->vel.velY = 100*((float)rand()/RAND_MAX);		
		p->vel.velZ = 100*((float)rand()/RAND_MAX);		
	}
}

/*update velocity w.r.t to common pattern () and then 
update position w.r.t formula given: p.x = p.x + v.x.dt where dt=1 */
void update_velocity_position_in_cpu(Particle *p)
{
	struct timeval start_time;
	struct timeval stop_time;

	gettimeofday(&start_time, NULL);
	for(int j=0; j<NUM_ITERATIONS; j++)
	{
		for(int i=0; i<NUM_PARTICLES; i++)
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

void update_velocity_position_rand_in_cpu(Particle *p)
{
        struct timeval start_time;
        struct timeval stop_time;

        gettimeofday(&start_time, NULL);
        for(int j=0; j<NUM_ITERATIONS; j++)
        {
                for(int i=0; i<NUM_PARTICLES; i++)
                {
                        p[i].vel.velX = 100*((float)rand()/RAND_MAX);
                        p[i].vel.velY = 100*((float)rand()/RAND_MAX);
                        p[i].vel.velZ = 100*((float)rand()/RAND_MAX);

                        p[i].pos.posX = p[i].pos.posX + p[i].vel.velX;
                        p[i].pos.posY = p[i].pos.posY + p[i].vel.velY;
                        p[i].pos.posZ = p[i].pos.posZ + p[i].vel.velZ;
                }
        }
        gettimeofday(&stop_time, NULL);


        printf("Total time of Execution in CPU: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
}


__global__ void particle_kernel(Particle *p, int Nparticles, int Niterations)
{
	int i = (blockIdx.x*blockDim.x)+threadIdx.x;

        if(i < Nparticles) {
		for(int j=0; j<Niterations; j++)
		{
              		(i+j)%2 ? (p[i].vel.velX += (5*(i+j))%100) : (p[i].vel.velX -= (5*(i+j))%100);
                        (i+j)%2 ? (p[i].vel.velY += (3*(i+j))%100) : (p[i].vel.velY -= (3*(i+j))%100);
                        (i+j)%2 ? (p[i].vel.velZ += (7*(i+j))%100) : (p[i].vel.velZ -= (7*(i+j))%100);

                        p[i].pos.posX = p[i].pos.posX + p[i].vel.velX;
                        p[i].pos.posY = p[i].pos.posY + p[i].vel.velY;
                        p[i].pos.posZ = p[i].pos.posZ + p[i].vel.velZ;
                }
        }
	__syncthreads();
}

__global__ void particle_kernel_per_iteration(Particle *p, int Nparticles)
{
        int i = (blockIdx.x*blockDim.x)+threadIdx.x;

        if(i < Nparticles) {
		p[i].pos.posX = p[i].pos.posX + p[i].vel.velX;
             	p[i].pos.posY = p[i].pos.posY + p[i].vel.velY;
               	p[i].pos.posZ = p[i].pos.posZ + p[i].vel.velZ;
        }
        __syncthreads();
}


void update_velocity_position_in_gpu(Particle *p)
{
        struct timeval start_time;
        struct timeval stop_time;


	Particle *gPar = NULL;	
	cudaMalloc(&gPar, NUM_PARTICLES*sizeof(Particle));

        gettimeofday(&start_time, NULL);

        cudaMemcpy(gPar, p, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
	particle_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(gPar, NUM_PARTICLES, NUM_ITERATIONS);
	cudaDeviceSynchronize();
	cudaMemcpy(p, gPar, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);

        gettimeofday(&stop_time, NULL);
	
        printf("Total time of Execution in GPU: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
	cudaFree(gPar);
}

void update_velocity_position_rand_in_gpu(Particle *p)
{
        struct timeval start_time;
        struct timeval stop_time;


	Particle *gPar = NULL;	
	cudaMalloc(&gPar, NUM_PARTICLES*sizeof(Particle));

        gettimeofday(&start_time, NULL);

        for(int i=0; i<NUM_ITERATIONS; i++)
        {
        	cudaMemcpy(gPar, p, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

        	p[i].vel.velX = 100*((float)rand()/RAND_MAX);
                p[i].vel.velY = 100*((float)rand()/RAND_MAX);
                p[i].vel.velZ = 100*((float)rand()/RAND_MAX);
		
		particle_kernel_per_iteration<<<GRID_SIZE, BLOCK_SIZE>>>(gPar, NUM_PARTICLES);
		cudaDeviceSynchronize();

		cudaMemcpy(p, gPar, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
        }

        gettimeofday(&stop_time, NULL);
	
        printf("Total time of Execution in GPU: %ld usec\n\n",
                (stop_time.tv_sec*1000000 + stop_time.tv_usec)-(start_time.tv_sec*1000000 + start_time.tv_usec));
	cudaFree(gPar);
}

int main(int argc, char *argv[])
{
	int input = 0;

	if(argc != 3)
	{
		printf("No. of arguments to be passed should be 2\n");
		exit(1);
	}

	NUM_PARTICLES = atoi(argv[1]);
	BLOCK_SIZE = atoi(argv[2]);
	
	Particle *par = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
	fill_data(par);

	printf("Enter 1 for CPU(with rand), 2 for CPU(with pattern), 3 for GPU(with rand)/CPU dependency, 4 for GPU(with pattern)\n");
	fflush(stdout);
	scanf("%d",&input);

	switch(input) 
	{
		case 1: update_velocity_position_rand_in_cpu(par);
			break;

		case 2: update_velocity_position_in_cpu(par);
			break;

		case 3: update_velocity_position_rand_in_gpu(par);
			break;
 
		case 4: update_velocity_position_in_gpu(par);
			break;

		default: printf("Wrong Input\n");
			 break;
	} 
	
	free(par);
	return 0;
}

