#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>



__global__ void initializePagerankArray(float * pagerank_d, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n_vertices) {
        pagerank_d[i] = 1.0/(float)n_vertices;
    }
}

__global__ void setPagerankNextArray(float * pagerank_next_d, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < n_vertices) {
        pagerank_next_d[i] = 0.0;
    }
}


__global__ void addToNextPagerankArray(float * pagerank_d, float * pagerank_next_d, int * n_successors_d, int * successors_d, int * successor_offset_d, float * dangling_value2, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j;
    int n_suc;

    if (i < n_vertices) {
        n_suc = n_successors_d[i];
        if(n_suc > 0) {
            for(j = 0; j < n_suc; j++) {
                atomicAdd(&(pagerank_next_d[successors_d[successor_offset_d[i]+j]]), 0.85*(pagerank_d[i])/n_suc);
            }
        } else {
            atomicAdd(dangling_value2, 0.85*pagerank_d[i]);
        }
    }
}       


__global__ void finalPagerankArrayForIteration(float * pagerank_next_d, int n_vertices, float dangling_value2) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;


    if(i < n_vertices) {
        pagerank_next_d[i] += (dangling_value2 + (1-0.85))/((float)n_vertices);
    }
}


__global__ void setPagerankArrayFromNext(float * pagerank_d, float * pagerank_next_d, int n_vertices) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(i < n_vertices) {
        pagerank_d[i] = pagerank_next_d[i];
        pagerank_next_d[i] = 0.0;
    }
}      

__global__ void convergence(float * pagerank_d, float * pagerank_next_d, float * reduced_sums_d, int n_vertices) {
// Each thread computes the diff for two vertexes (thus, half # of blocks needed for this function)
// Because of this, we need to handle the case where only one block is needed
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int i_thr = threadIdx.x;

    __shared__ float sums[1024];                       // blockDim.x == 1024
    
    float temp1, temp2;    
    
    if(i < 1024) {
        reduced_sums_d[i] = 0;
    }

    if(i < n_vertices) {
        temp1 = pagerank_next_d[i] - pagerank_d[i];
        if(i + (1024 * gridDim.x) < n_vertices) {
            temp2 = pagerank_next_d[i+ (1024 * gridDim.x)] - pagerank_d[i +(1024*gridDim.x)];
        }else{
            temp2 = 0;
        }

        if(temp1 < 0) {
            temp1 = temp1 * (-1);
        }
        if(temp2 < 0) {
            temp2 = temp2 * (-1);
        }

        sums[i_thr] = temp1 + temp2;
    } else {
        sums[i_thr] = 0;
    }
    __syncthreads();

    int j, index, index2;
    index = i_thr;

    for(j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
        if((index+1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
            index2 = index - (1 << j);
            sums[index] += sums[index2];
        }
        __syncthreads();
    }

    reduced_sums_d[blockIdx.x] = sums[1023];
}

__global__ void getConvergence(float * reduced_sums_d, float * diff) {
    int j, index, index2;
    index = threadIdx.x;
    
    for(j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
        if((index+1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
            index2 = index - (1 << j);
            reduced_sums_d[index] += reduced_sums_d[index2];
        }
        __syncthreads();
    }    

    *diff = reduced_sums_d[1023]; 
}


int main(int argc, char ** args) {
    if (argc != 2) {
	fprintf(stderr,"Wrong number of args. Provide input graph file.\n");
        exit(-1);
    } 

    // Error code to check return values for CUDA calls
    cudaFree(0);   // Set the cuda context here so that when we time, we're not including initial overhead
    cudaError_t err = cudaSuccess;
    cudaProfilerStart();
    
    // Start CPU timer
    clock_t cycles_to_build, cycles_to_calc;
    clock_t start = clock();


/*************************************************************************/
    // build up the graph
    int i;
    unsigned int n_vertices = 0;
    unsigned int n_edges = 0;
    unsigned int vertex_from = 0, vertex_to = 0, vertex_prev = 0;
    
    // Flattened data structure variables
    float * pagerank_h, *pagerank_d;
    float *pagerank_next_d;
    int * n_successors_h, *n_successors_d;
    int * successors_h, *successors_d;                // use n_edges to initialize
    int * successor_offset_h, *successor_offset_d;

    FILE * fp;
    if ((fp = fopen(args[1], "r")) == NULL) {
        fprintf(stderr,"ERROR: Could not open input file.\n");
        exit(-1);
    }

    // parse input file to count the number of vertices
    // expected format: vertex_from vertex_to
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        if (vertex_from > n_vertices) {
            n_vertices = vertex_from;
	}
        else if (vertex_to > n_vertices) {
            n_vertices = vertex_to;
	}
	n_edges++;
    }
    n_vertices++;
    
 
    // Allocate flattened data structure host and device memory
    pagerank_h = (float *) malloc(n_vertices * sizeof(*pagerank_h));
    err = cudaMalloc((void **)&pagerank_d, n_vertices*sizeof(float));
    err = cudaMalloc((void **)&pagerank_next_d, n_vertices*sizeof(float));
    n_successors_h = (int *) calloc(n_vertices, sizeof(*n_successors_h));
    err = cudaMalloc((void **)&n_successors_d, n_vertices*sizeof(int));
    successor_offset_h = (int *) malloc(n_vertices * sizeof(*successor_offset_h));
    err = cudaMalloc((void **)&successor_offset_d, n_vertices*sizeof(int));


    // Allocate memory for contiguous successors_d data
    successors_h = (int *) malloc(n_edges * sizeof(*successors_h));
    err = cudaMalloc((void **)&successors_d, n_edges*sizeof(int));

    // allocate memory for successor pointers
    int offset = 0, edges = 0;                      // offset into the successors_h array

    // parse input file to count the number of successors of each vertex
    fseek(fp, 0L, SEEK_SET);
    i = 0;
 
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        n_successors_h[vertex_from] += 1;
	
	// Fill successor_offset_h array
        successor_offset_h[i] = offset;
	if(edges != 0 && vertex_prev != vertex_from) {
	    i = vertex_from;
	    offset = edges;
	    successor_offset_h[i] = offset;
	   
	    vertex_prev = vertex_from;
	}

	// Fill successor array
	successors_h[edges] = vertex_to;
	
	edges++;
    }
    successor_offset_h[i] = edges - 1;    

    fclose(fp);

    // Get build time and reset start
    cycles_to_build = clock() - start;
    start = clock();

/**************************************************************/
    // Transfer data structure to the GPU
    err = cudaMemcpy(n_successors_d, n_successors_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successors_d, successors_h, n_edges*sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpy(successor_offset_d, successor_offset_h, n_vertices*sizeof(int), cudaMemcpyHostToDevice);

/*************************************************************************/
    // Compute the pagerank
    int n_iterations = 30;
    int iteration = 0;
    int numOfBlocks = 1;                          // default example value for 1000 vertex graph
    int threadsPerBlock = 1000;                   // default example value for 1000 vertex graph

    if(n_vertices <= 1024) {
        threadsPerBlock = n_vertices;
        numOfBlocks = 1;
    } else {
        threadsPerBlock = 1024;
        numOfBlocks = (n_vertices + 1023)/1024;   // The "+ 1023" ensures we round up
    }

    float dangling_value_h = 0;
    float dangling_value_h2 = 0;
    float *dangling_value2, *reduced_sums_d;
    int n_blocks = (n_vertices + 2048 - 1)/2048;
    if (n_blocks == 0){
        n_blocks = 1;
    }
    float epsilon = 0.000001;
    float * d_diff;
    float h_diff = epsilon + 1;

    err = cudaMalloc((void **)&d_diff, sizeof(float));
    err = cudaMalloc((void **)&reduced_sums_d, 1024 * sizeof(float));     
    err = cudaMalloc((void **)&dangling_value2, sizeof(float));
    err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);

    initializePagerankArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, n_vertices);
    setPagerankNextArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_next_d, n_vertices);
    cudaDeviceSynchronize();
    
    while(epsilon < h_diff && iteration < n_iterations) {  //was 23
       // set the dangling value to 0 
        dangling_value_h = 0;
        err = cudaMemcpy(dangling_value2, &dangling_value_h, sizeof(float), cudaMemcpyHostToDevice);
       
        // initial parallel pagerank_next computation
        addToNextPagerankArray<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, pagerank_next_d, n_successors_d, successors_d, successor_offset_d, dangling_value2, n_vertices);

        // get the dangling value
        err = cudaMemcpy(&dangling_value_h2, dangling_value2, sizeof(float), cudaMemcpyDeviceToHost); 

        // final parallel pagerank_next computation
        finalPagerankArrayForIteration<<<numOfBlocks,threadsPerBlock>>>(pagerank_next_d, n_vertices, dangling_value_h2);

        // Test for convergence
        convergence<<<n_blocks,1024>>>(pagerank_d, pagerank_next_d, reduced_sums_d, n_vertices); 
        getConvergence<<<1,1024>>>(reduced_sums_d, d_diff);
        
        // Get difference to compare to epsilon
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);      

        // Make pagerank_d[i] = pagerank_next_d[i]
        setPagerankArrayFromNext<<<numOfBlocks,threadsPerBlock>>>(pagerank_d, pagerank_next_d, n_vertices);
        cudaDeviceSynchronize();

        iteration++;
    }


    //****** transfer pageranks from device to host memory ************************************************ 
    err = cudaMemcpy(pagerank_h, pagerank_d, n_vertices*sizeof(float), cudaMemcpyDeviceToHost);

    // Find CPU elapsed time
    cycles_to_calc = clock() - start;
    
    // Print time taken
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
 
    // Print pageranks
   // for(i = 0; i < n_vertices; i++) {
   //     printf("i: %d, pr: %.6f\n",i, pagerank_h[i]);
   // }    

    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);
    printf("iter: %d\n", iteration);

// Tests: ***************************
/*    int n_vert = 1000;    

    n_blocks = (n_vert + 2048 - 1)/2048;
    if(n_blocks == 0){
        n_blocks = 1;
    } 

    float * h_A = (float *) malloc(n_vert * sizeof(float));
    float * h_B = (float *) malloc(n_vert * sizeof(float));
    float * d_A = NULL;
    float * d_B = NULL;
    float * d_C = NULL;
    err = cudaMalloc((void **)&d_A, n_vert * sizeof(float));
    err = cudaMalloc((void **)&d_B, n_vert * sizeof(float)); 
    err = cudaMalloc((void **)&d_C, 1024 * sizeof(float));     

    for(i = 0; i < n_vert; i++) {
        h_A[i] = 1;
        h_B[i] = 1.0009;
    }
    err = cudaMemcpy(d_A, h_A, n_vert *sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, n_vert*sizeof(float), cudaMemcpyHostToDevice);

    convergence<<<n_blocks,1024>>>(d_A, d_B, d_C, n_vert); 
    cudaDeviceSynchronize();

  //  float * d_diff;
   // float h_diff;
    err = cudaMalloc((void **)&d_diff, sizeof(float));

     getConvergence<<<1,1024>>>(d_C, d_diff);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
    printf("this: %.3f\n", h_diff);
     */

// End Tests ************************

    // Free device global memory
    err = cudaFree(pagerank_d);
    err = cudaFree(pagerank_next_d);

    // Free host memory
    free(pagerank_h);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

