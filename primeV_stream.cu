
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */



#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>


#define HANDLE_ERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


/*#define N   10000
#define FULL_DATA_SIZE   (N*100)*/


/*__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;
        float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
        c[idx] = (as + bs) / 2;
    }
}*/

__global__ void primeP_gpu (int max, int *A, int *count, int cnt)
{
 
    int n = blockDim.x * blockIdx.x + threadIdx.x;
 
    // do nothing if we are not in the useable space of
    // threads (see kernel launch call: you may be creating
    // more threads than you need)
    if (n >= max) return;
 
    unsigned int a = A[n];
    int i;
 
    for (i = 2; i < a; i++)
    {
    	if (a % i == 0 && i != a)
    		break;
    }
 
    if (a == i) {
        // don't do this: threads overwrite each other's values
        // causing undercount:
        //    *count = *count + 1;
 
        // instead, use atomic operations:
        atomicAdd(&count[cnt], 1);
    }
 
}

int main( int argc, char *argv[] ) {

	cudaEvent_t     start, stop;
    float           elapsedTime;

    cudaStream_t    stream0, stream1;
    int *host_a, *host_c;
    int *dev_a0;
    int *dev_a1;
    int *dev_c0;
    int *dev_c1;    
    
	char *filename;
	if(argc < 2)
	{
		printf("too few arguments");
		return 0;
	}
	filename = (char *) malloc(strlen(argv[1]));
	strcpy(filename, argv[1]);
	
	FILE *file;  
	char *input;
	size_t len = 0;
	int i=0;
	int maxTested = 100000000;
	
	// allocate host locked memory, used to stream
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_a,
                              maxTested * sizeof(int),
                              cudaHostAllocDefault ) );

	                   
    file = fopen(filename, "r");
	while(!feof(file))
	{
		getline(&input, &len, file);
		host_a[i++] = atoi(input);
	}
	
	int data_size = i;
	int N = data_size/100;	
	
    cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

    

    // start the timers
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );

    // initialize the streams
    HANDLE_ERROR( cudaStreamCreate( &stream0 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream1 ) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a0,
                              N * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a1,
                              N * sizeof(int) ) );
                              
	
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c0,
                              100 * sizeof(int) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c1,
                              100 * sizeof(int) ) );
	HANDLE_ERROR( cudaMemset( (void*)dev_c0, 0,
                              100 * sizeof(int) ) );
    HANDLE_ERROR( cudaMemset( (void*)dev_c1, 0,
                              100 * sizeof(int) ) ); 							                                
    
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_c,
                              100 * sizeof(int),
                              cudaHostAllocDefault ) );

    
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    
    unsigned int threads_per_block = 1024;
    unsigned int num_blocks = ceil (N / (1.0*threads_per_block) );

    // now loop over full data, in bite-sized chunks
    int cnt =0;
    for (int i=0; i<data_size; i+= N*2) {

        // enqueue copies of a and b for stream0
        HANDLE_ERROR( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );

        // enqueue kernel in stream0
        primeP_gpu<<<num_blocks,threads_per_block,0,stream0>>>( N, dev_a0, dev_c0, cnt );

        // enqueue copies of a and b for stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );

        // enqueue kernel in stream1
        primeP_gpu<<<num_blocks,threads_per_block,0,stream1>>>( N, dev_a1, dev_c1, cnt);

        // enqueue copies of c from device to locked memory
        HANDLE_ERROR( cudaMemcpyAsync( host_c+cnt, dev_c0+cnt,
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( host_c+cnt+1, dev_c1+cnt,
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
        cnt +=2;
    }

    HANDLE_ERROR( cudaStreamSynchronize( stream0 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );
    
    int h_numPrimes=0;
    for(int i=0;i<100;i++)
    {
    	h_numPrimes += host_c[i];
    }

    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );

    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
                                        
    printf( "Time taken:  %3.1f s\n", elapsedTime/1000 );
    printf("Number of primes: %d\n", h_numPrimes);

    // cleanup the streams and memory
    HANDLE_ERROR( cudaFreeHost( host_a ) );
    HANDLE_ERROR( cudaFree( dev_a0 ) );
    HANDLE_ERROR( cudaFree( dev_a1 ) );
    
    HANDLE_ERROR( cudaFreeHost( host_c ) );
    HANDLE_ERROR( cudaFree( dev_c0 ) );
    HANDLE_ERROR( cudaFree( dev_c1 ) );
    
    HANDLE_ERROR( cudaStreamDestroy( stream0 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );

    return 0;
}
