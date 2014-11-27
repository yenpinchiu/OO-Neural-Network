import math,random

class edge:

    def __init__(self):
        self.weight = None
        self.from_node = None
        self.to_node = None

class node:

    def __init__(self):

        self.value = None
        self.delta = None
        self.in_edge = []
        self.out_edge = []
        self.bias = None

    def build_in_edge(self,linked_node):
        new_edge = edge()
        new_edge.from_node = linked_node
        new_edge.to_node = self
        self.in_edge.append(new_edge)
        linked_node.out_edge.append(new_edge)

    def build_out_edge(self,linked_node):
        new_edge = edge()
        new_edge.from_node = self
        new_edge.to_node = linked_node
        self.out_edge.append(new_edge)
        linked_node.in_edge.append(new_edge)

def sigmoid(x):
    return 1/(1+math.exp(-x))

class network:

    def __init__(self,*layers):

        self.layers = []
        self.nodes = []

        for layer_id in range(0,len(layers)):
            self.layers.append([])
            for node_id in range(0,layers[layer_id]):
                new_node = node()
                self.nodes.append(new_node)
                self.layers[layer_id].append(new_node)
                if layer_id != 0:
                    for previous_layer_node in self.layers[layer_id-1]:
                        new_node.build_in_edge(previous_layer_node)

    def init_random_weights(self):
        for node in self.nodes:
            for edge in node.in_edge:
                if edge.weight == None:  
                    edge.weight = random.uniform(-1.0,1.0)
            for edge in node.out_edge:
                if edge.weight == None:
                    edge.weight = random.uniform(-1.0,1.0)
            if node not in self.layers[0]:
                node.bias = random.uniform(-1.0,1.0)
  
    def back_propagation_propagate(self,trainning_data_x):
        
        for input_layer_node_id in range(0,len(self.layers[0])):
            self.layers[0][input_layer_node_id].value = trainning_data_x[input_layer_node_id]
        
        for hiddenlayer in self.layers[1:]:
            for hiddenlayer_node in hiddenlayer:
                tmp_sum = 0
                for edge in hiddenlayer_node.in_edge:
                    tmp_sum += edge.from_node.value*edge.weight
                tmp_sum += hiddenlayer_node.bias
                hiddenlayer_node.value = sigmoid(tmp_sum)

    def back_propagation_error(self,trainning_data_y):
      
        mse = 0
        for output_node_id in range(0,len(self.layers[len(self.layers)-1])):
            mse += 0.5*(trainning_data_y[output_node_id] - self.layers[len(self.layers)-1][output_node_id].value)**2
        return mse
    
    def back_propagation_delta(self,trainning_data_y):
        for layer_id in range(len(self.layers)-1,-1,-1):
            for node_id in range(0,len(self.layers[layer_id])):
                node = self.layers[layer_id][node_id]
                if layer_id == len(self.layers)-1:
                    node.delta = (trainning_data_y[node_id] - node.value)* node.value*(1- node.value)
                else:
                    node.delta = 0
                    for edge in node.out_edge:
                        node.delta += edge.weight*edge.to_node.delta
                    node.delta = node.delta*node.value*(1-node.value)
    
    def back_propagation_update_weight(self,trainning_data_y,learning_rate):
        for layer_id in range(len(self.layers)-1,-1,-1):
            for node_id in range(0,len(self.layers[layer_id])):
                node = self.layers[layer_id][node_id]
                for edge in node.in_edge:
                    edge.weight += learning_rate*node.delta*edge.from_node.value
                if node.bias != None:
                    node.bias += learning_rate*node.delta

    def back_propagation_train(self,trainning_data,iteration):

        for iter_id in range(0,iteration):
            iter_mse = 0
            for single_data in trainning_data:
                trainning_data_x = single_data[0]
                trainning_data_y = single_data[1]
                self.back_propagation_propagate(trainning_data_x)
                iter_mse += self.back_propagation_error(trainning_data_y)
                self.back_propagation_delta(trainning_data_y)
                self.back_propagation_update_weight(trainning_data_y,0.5)
            print("iteration " + str(iter_id) +" mse:"+str(iter_mse))     
        print("trainning done")
    
    def back_propagation_predict(self,trainning_data):
        predict_data = []
        for single_data in trainning_data:
            trainning_data_x = single_data[0]
            predict_data_y = []
            self.back_propagation_propagate(trainning_data_x)
            for node_id in range(0,len(self.layers[len(self.layers)-1])):
                predict_data_y.append(self.layers[len(self.layers)-1][node_id].value)
            predict_data.append([trainning_data_x,predict_data_y])
        return predict_data
            
if __name__ == '__main__':

    trainning_data = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    n = network(2,2,1)
    n.init_random_weights()
    n.back_propagation_train(trainning_data,3000)
    predict_data = n.back_propagation_predict(trainning_data)

    for single_data in predict_data:
        print(single_data)

