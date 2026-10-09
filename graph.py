# this implements graph based layer traversal

from main import BareboneNN
import numpy as np

class BareboneNNGraph(BareboneNN):
    def __init__(self):
        self.loss_function = None
        self.nodes = []
        self.layers_list:list[BaseLayerNode] = []

    def __topological_sort(self,all_nodes):
        """
        Kahn's Algorithm for Topological Sort.
        :param all_nodes: A list or collection of all nodes in the graph.
        :return: A list representing the topologically sorted order, or an empty list if a cycle exists.
        """
        in_degree = {node: 0 for node in all_nodes}

        # Step 1: Calculate in-degrees for all nodes
        for node in all_nodes:
            for conn in node.connections:
                in_degree[conn] += 1

        # Step 2: Initialize queue with all nodes that have an in-degree of 0
        q = deque([node for node in all_nodes if in_degree[node] == 0])

        topological_order = []

        # Step 3: Process the queue
        while q:
            node = q.popleft()
            topological_order.append(node)

            # "Remove" outgoing edges by decrementing neighbor in-degrees
            for conn in node.connections:
                in_degree[conn] -= 1
                # If in-degree becomes 0, add it to the queue
                if in_degree[conn] == 0:
                    q.append(conn)

        # Step 4: Cycle detection check
        if len(topological_order) != len(all_nodes):
            print("Error: Graph contains a cycle! Topological sort not possible.")
            return []

        return topological_order

    def add_layers(self,*layer):
        self.nodes.extend(layer)

    def connect(self,layer1:BaseLayerNode,layer2:BaseLayerNode):
        # make the connection
        layer1.connect(layer2)

    def freeze(self):
        # Required for graph based one it creates the graph
        self.layers_list = self.__topological_sort(self.nodes)

    def __train_batch(self, batch_x, batch_y):
        # -------------------------------------------------------------
        # 1. FORWARD PASS (Topological Order)
        # -------------------------------------------------------------
        for index, layer in enumerate(self.layers_list):
            upstream_nodes = list(layer.upstream.keys())

            if index == 0:
                # First layer takes the batch input directly
                layer.data = batch_x
            elif len(upstream_nodes) == 1:
                # Single input connection
                layer.data = upstream_nodes[0].forward_data
            else:
                # Multiple input connections: concatenate features along axis 1
                layer.data = np.hstack([node.forward_data for node in upstream_nodes])

            # Compute forward activation
            layer.forward()

        # Final output layer data
        output_layer = self.layers_list[-1]
        forward_data = output_layer.forward_data

        # -------------------------------------------------------------
        # 2. BACKWARD PASS (Reverse Topological Order)
        # -------------------------------------------------------------
        # Dictionary to store incoming errors accumulated for each node
        # Key: node, Value: list of error matrices from downstream connections
        incoming_errors = {node: [] for node in self.layers_list}

        # Initial loss gradient at the final layer
        loss_error = self.loss_function.get_errors(forward_data, batch_y)
        incoming_errors[output_layer].append(loss_error)

        for layer in reversed(self.layers_list):
            # Step A: Sum all error matrices arriving from downstream connections
            total_layer_error = sum(incoming_errors[layer])

            # Step B: Backpropagate error through layer's own weights/activation
            # err_total shape: (Batch_Size, total_features_in)
            err_total = layer.backward(total_layer_error)

            # Step C: Route error to upstream nodes
            upstream_nodes = list(layer.upstream.keys())
            if not upstream_nodes:
                continue

            if len(upstream_nodes) == 1:
                # Direct 1-to-1 error routing
                incoming_errors[upstream_nodes[0]].append(err_total)
            else:
                # Multi-input node: Split column-wise based on feature sizes of upstream nodes
                col_sizes = [node.forward_data.shape[1] for node in upstream_nodes]
                split_indices = np.cumsum(col_sizes)[:-1]
                splitted_errors = np.hsplit(err_total, split_indices)

                # Assign corresponding feature error slice to each upstream node
                for ups_node, err_slice in zip(upstream_nodes, splitted_errors):
                    incoming_errors[ups_node].append(err_slice)
        
        return loss_error
                    

                    



            

                    

                    

            



# from layers import *
# from loss_function import CrossEntropyBinaryLoss
# from collections import deque
        

# model = BareboneNNGraph()

# inp = LinearLayer(10,20,0.01)

# branch1 = LinearLayer(20,64,0.05)
# branch2 = LinearLayer(20,64,0.05)

# hidden = ReluLayer()

# comb = LinearLayer(64,1,0.04)
# out = SigmoidLayer()

# model.add_layers(
#     inp,
#     branch1,branch2,
#     hidden,
#     comb,
#     out
# )

# model.connect(inp,branch1)
# model.connect(inp,branch2)
# model.connect(branch1,hidden)
# model.connect(branch2,hidden)
# model.connect(hidden,comb)
# model.connect(comb,out)

# model.loss_function = CrossEntropyBinaryLoss()
# model.freeze()

# model.fit()





