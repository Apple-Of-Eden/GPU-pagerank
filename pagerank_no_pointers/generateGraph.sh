# Will output a random graph to be used as input to the pagerank code 
if [ $# -lt 2 ]
  then
    echo "No arguments supplied. Give: [# of vertices] [max # of edges per vertex]"
    exit 1
fi

rm randomGraph.txt                                 

for ((i=0; i < $1; i++))
do
    hasOutgoing=$(shuf -i0-3 -n1)   # Vertex has a 3/4 chance of having an outgoing edge
    if [ $hasOutgoing -gt 0 ] || [ $i == $(($1 - 1)) ];
        then
        numOfEdges=$(shuf -i1-$2 -n1)
            
        for ((j = 0; j < $numOfEdges; j++))
        do
            edgeTo=$(shuf -i0-$(($1 - 1)) -n1)         # edgeTo is from 0 to $1 - 1 
            if [ $edgeTo -ne $i ];                     # Don't allow for edge to self
                then
                lineText="$i $edgeTo"
                echo $lineText >> randomGraph.txt
            fi
        done
    fi
done
