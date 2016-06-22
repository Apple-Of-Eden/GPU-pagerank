#include <stdio.h>
#include <time.h>

typedef struct vertex vertex;

struct vertex {
    unsigned int vertex_id;
    float pagerank;
    float pagerank_next;
    unsigned int n_successors;
    unsigned int * all_successors;
};

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
    int i,j;
    unsigned int n_vertices = 0;
    unsigned int vertex_from = 0, vertex_to = 0;

    vertex * vertices;

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
    }
    n_vertices++;

    // allocate memory for vertices
    vertices = (vertex *)malloc(n_vertices*sizeof(vertex));
    if (!vertices) {
        fprintf(stderr,"Malloc failed for vertices.\n");
        exit(-1);
    }
    memset((void *)vertices, 0, (size_t)(n_vertices*sizeof(vertex)));

    // parse input file to count the number of successors of each vertex
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%u %u", &vertex_from, &vertex_to) != EOF) {
        vertices[vertex_from].n_successors++;
    }

    // allocate memory for successor pointers
    for (i=0; i<n_vertices; i++) {
        vertices[i].vertex_id = i;
        if (vertices[i].n_successors > 0) {
	    vertices[i].all_successors = (unsigned int *)calloc(vertices[i].n_successors,sizeof(unsigned int));
        }
    }

    // parse input file to set up the successor pointers
    fseek(fp, 0L, SEEK_SET);
    while (fscanf(fp, "%d %d", &vertex_from, &vertex_to) != EOF) {	
	for(i = 0; i < vertices[vertex_from].n_successors; i++) {
	    if(vertices[vertex_from].all_successors[i] == 0) {
		vertices[vertex_from].all_successors[i] = vertex_to + 1; //+1 so we can allow 0 to be the NULL index value
		break;
	    }
	}	
    }

    fclose(fp);

    cycles_to_build = clock() - start;
    start = clock();


    /*************************************************************************/
    // compute the pagerank

    unsigned int n_iterations = 24;
    float alpha = 0.85;
    float eps   = 0.000001;

    // run on the host
    unsigned int i_iteration;

    float value, diff;
    float pr_dangling_factor = alpha / (float)n_vertices;   // pagerank to redistribute from dangling nodes
    float pr_dangling;
    float pr_random_factor = (1-alpha) / (float)n_vertices; // random portion of the pagerank
    float pr_random;
    float pr_sum, pr_sum_inv, pr_sum_dangling;
    float temp;

    // initialization
    for (i=0;i<n_vertices;i++) {
        vertices[i].pagerank = 1 / (float)n_vertices;
        vertices[i].pagerank_next =  0;
    }

    pr_sum = 0;
    pr_sum_dangling = 0;
    for (i=0; i<n_vertices; i++) {
        pr_sum += vertices[i].pagerank;
        if (!vertices[i].n_successors) {
            pr_sum_dangling += vertices[i].pagerank;
	}
    }

    i_iteration = 0;
    diff = eps+1;

    while ( (diff > eps) && (i_iteration < n_iterations) ) { // can do 23 iterations for 1 million nodes
        for (i=0;i<n_vertices;i++) {
            if (vertices[i].n_successors) {
                value = (alpha * vertices[i].pagerank)/vertices[i].n_successors; //value = vote split equally
            } else {
                value = 0;
            }

            for (j=0;j<vertices[i].n_successors;j++) {               // pagerank_next = sum of votes linking to it
                //vertices[i].successors[j]->pagerank_next += value;
		vertices[vertices[i].all_successors[j]-1].pagerank_next += value;
            }
        }
   
        // for normalization
        pr_sum_inv = 1/pr_sum;

        // alpha
        pr_dangling = pr_dangling_factor * pr_sum_dangling;
        pr_random = pr_random_factor * pr_sum;

        pr_sum = 0;
        pr_sum_dangling = 0;

        diff = 0;
        for (i=0;i<n_vertices;i++) {
            // update pagerank
            temp = vertices[i].pagerank;
            vertices[i].pagerank = vertices[i].pagerank_next*pr_sum_inv + pr_dangling + pr_random;
            vertices[i].pagerank_next = 0;

            // for normalization in next cycle
            pr_sum += vertices[i].pagerank;
            if (!vertices[i].n_successors)
                pr_sum_dangling += vertices[i].pagerank;

            // convergence
            diff += abs_float(temp - vertices[i].pagerank);
           // printf("prev: %.12f, pr: %.12f\n",temp, vertices[i].pagerank);
        }
       // printf("Iteration %u:\t diff = %.12f\n", i_iteration, diff);

        i_iteration++;
   }
 /*************************************************************************/
    // End CPU Timer
    cycles_to_calc = clock() - start;
    
    // Print time
    int build_msec = cycles_to_build * 1000 / CLOCKS_PER_SEC;
    int calc_msec = cycles_to_calc * 1000 / CLOCKS_PER_SEC;

    // print pageranks
    for (i=0;i<n_vertices;i++) {
        //printf("Vertex %u:\tpagerank = %.6f\n", i, vertices[i].pagerank);
    }
 
    printf("Time to build: %d seconds, %d milliseconds\n", build_msec/1000, build_msec%1000);
    printf("Time to calc: %d seconds, %d milliseconds\n", calc_msec/1000, calc_msec%1000);
 
    printf("iter: %d\n",i_iteration);

    printf("Done\n");
    return 0;
}

