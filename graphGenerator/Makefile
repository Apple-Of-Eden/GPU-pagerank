all: genGraph

genGraph: genGraph.o
	gcc genGraph.o -o genGraph

genGraph.o:genGraph.c
	gcc -Wall -O -c genGraph.c

clean:
	rm -f genGraph genGraph.o

