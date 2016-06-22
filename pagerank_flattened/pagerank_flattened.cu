#include <stdio.h>
#include <time.h>


float abs_float(float in) {
  if (in >= 0)
    return in;
  else
    return -in;
}


int main(int argc, char ** args) {
    if (argc != 2) {
	fprintf(stderr,"Wrong number of args. Provide input graph file.\n");
        exit(-1);
    }    

    // Start CPU timer
    clock_t cycles_to_build, cycles_to_calc;
    clock_t start = clock();

    // build up the graph
    int i;
    unsigned int n_vertices = 0;
    unsigned int n_edges = 0;
    unsigned int vertex_from = 0, vertex_to = 0, vertex_prev = 0;
    
    // Flattened data structure variables
    float * pagerank_h;
    float *pagerank_next_h;
    int * n_successors_h;
    int * successors_h;                // use n_edges to initialize
    int * successor_offset_h;

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
    pagerank_next_h = (float *) malloc(n_vertices * sizeof(float));
    n_successors_h = (int *) calloc(n_vertices, sizeof(*n_successors_h));
    successor_offset_h = (int *) malloc(n_vertices * sizeof(*successor_offset_h));

    // Allocate memory for contiguous successors_d data
    successors_h = (int *) malloc(n_edges * sizeof(*successors_h));

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
   

/*************************************************************************/
    // Compute the pagerank
    int n_iterations = 30;
    int iteration = 0;
    float dangling_value_h = 0;
    float epsilon = 0.000001;
    float h_diff = epsilon + 1;

    for(int i = 0; i < n_vertices; i++) {
	pagerank_next_h[i] = 0;
	pagerank_h[i] = 1.0/(float)n_vertices;
    }

    int j, n_suc;
    while(epsilon < h_diff && iteration < n_iterations) {  
      // reset dangling_value_h and h_diff
       dangling_value_h = 0;
       h_diff = 0;
       
       // initial parallel pagerank_next computation
       for(i = 0; i < n_vertices; i++) {
	   n_suc = n_successors_h[i];
           if(n_suc > 0) {
               for(j = 0; j < n_suc; j++) {
                   pagerank_next_h[successors_h[successor_offset_h[i]+j]] += 0.85*(pagerank_h[i])/n_suc;
               }
           } else {
            	dangling_value_h += 0.85*pagerank_h[i];
           }
       }

        // final parallel pagerank_next computation
	for(i = 0; i < n_vertices; i++) {
            pagerank_next_h[i] += (dangling_value_h + (1-0.85))/((float)n_vertices);
    	}
	
        // Test for convergence
        for(i = 0; i < n_vertices; i++) {
    	    h_diff += abs_float(pagerank_next_h[i] - pagerank_h[i]);
        }
        
        // Make pagerank_d[i] = pagerank_next_d[i]
	for(i = 0; i < n_vertices; i++) {
            pagerank_h[i] = pagerank_next_h[i];
            pagerank_next_h[i] = 0.0;
    	}        
	
        iteration++;
    }


    // Find CPU elapsed time
    cycles_to_calc = clock() - start;
    
    // Print time taken
    int build_milli = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_milli = cycles_to_calc * 1000 / CLOCKS_PER_SEC;
 
    // Print pageranks
    //for(i = 0; i < n_vertices; i++) {
    //    printf("i: %d, pr: %.6f\n",i, pagerank_h[i]);
    //}    

    printf("Time to build: %d seconds, %d milliseconds\n",build_milli/1000, build_milli%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n",calc_milli/1000, calc_milli%1000);
    printf("iter: %d\n", iteration);


// ************************
    // Free host memory
    free(pagerank_h);

    printf("Done\n");
    return 0;
}

