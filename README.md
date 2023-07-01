# Graph_Neural_Network_Practice
 this is my code when learning graph neural network hope this helpful to someone who learning 
""if i describe something wrong am so sorry for that""

 ## what is graph?
 graph is the way to represent data that have a structure with a Vertex(Node) and edge. there are a way to represent a data like unweight(non-direction), weight and also bipatite graph. we can see the form of graph in a adjacency matix, edge_list, etc. in graph traditional method there are bag of degree, bag of graphlet, neighbor degree struture( degree is a node total number of connection), color refinement, etc all of these using static method to get graph information(node-level, edge-level, graph-level) .but the best way to learn represent of the graph is embedding and there are a method like node2vec and random walk but all of these is not looking into a node feature like random walk there random path and train to make it have a vector to close to each node if they have a short path so the new era of graph is a deeplearning method that really look into a node feature. this is really quick introduction from me if you interesting i reccommand to watch a lecture call cs224w by stanford in youtube this is really great free online source 

 ## Graph Convolutional Neural network
 ### simple_GCN(imprement on zachary karate club dataset)
 this process is a message passing(aggreation) and pass through a liner layer
 if these process repeat N time it will mean having N layer of GCN all of 
 these will output a embeding of Graph structure+feature and the last layer will be linear layer to classifire a node


 * first embeding before train process
 <img src="/image/simple_GCN_first_embed.png" alt="Alt text" title="Optional title" width="500" height="500">

 * last embeding after train process
 <img src="/image/simple_GCN_last_embed.png" alt="Alt text" title="Optional title" width="500" height="500">

 ## Graph Attention Neural network(upcoming)
 
