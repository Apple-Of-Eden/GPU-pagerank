#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

int main(int argc, char ** argv) {
    int v_from, v_to;
    FILE * fp;

    if(argc != 4) {
        printf("Wrong num of args. Need: [output filename] [# of vertices] [max # of edge per vertex]\n");
        return -1;
    }

    int fd = open(argv[1], O_TRUNC | O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        printf("open problem\n");
        return -1;
    }

    fp = fdopen(fd, "w");
    if(fp == NULL) {
        printf("uhoh\n");
        return -1;
    } 
    srand(time(NULL));

    int n_vertices = atoi(argv[2]);
    int max_edges = atoi(argv[3]);
    int i, j;
    int hasOutgoing, num_of_edges;
    
    for(i = 0; i < n_vertices; i++) {
        hasOutgoing = rand() % 4;                                // Chance of edge is 3/4
        if(hasOutgoing > 0 || i == 0 || i == n_vertices -1) {    // Check for edges (if first or last node, always make edge)
            num_of_edges = (rand() % max_edges) + 1;

            for(j = 0; j < num_of_edges;j++) {
                v_to = rand() % n_vertices;
                
                if (v_to != i) {                                 // Don't allow edge to self
                    fprintf(fp, "%d %d\n", i, v_to);
                }
            }
 
        }
    }

    return 1;
}

