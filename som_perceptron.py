import math,random

def sigmoid(x):
    return 1/(1+math.exp(-x))

class edge:

    def __init__(self):

        self.weight = None
        self.from_node = None
        self.to_node = None

class node:

    def __init__(self):

        self.value = None
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
    
class som_perceptron():

    def __init__(self,layers):

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
            node.bias = random.uniform(-1.0,1.0)

    def train(self,trainning_data,eta_att,eta_rep,echo):

        self.echo = echo
        self.eta_att = eta_att
        self.eta_rep = eta_rep

        for m in range(1,len(self.layers)):
            for echo_count in range(0,echo):
                print("layer " + str(m) + " echo " + str(echo_count), end="\t")
                same_class_longest_distance_pair = self.get_same_class_longest_distance_pair(trainning_data,m)
                different_class_shortest_distance_pair = self.get_different_class_shortest_distance_pair(trainning_data,m)
                self.update_weight(same_class_longest_distance_pair,different_class_shortest_distance_pair,m)   

    def update_weight(self,same_class_longest_distance_pair,different_class_shortest_distance_pair,layer_id):

        self.propagate(same_class_longest_distance_pair[0])
        y_p_m = self.get_layer_encoding(layer_id)
        y_p_m_1 = self.get_layer_encoding(layer_id-1)
        y_p_m_1.append(-1)
        
        self.propagate(same_class_longest_distance_pair[1])
        y_q_m = self.get_layer_encoding(layer_id)
        y_q_m_1 = self.get_layer_encoding(layer_id-1)
        y_q_m_1.append(-1) 

        self.propagate(different_class_shortest_distance_pair[0])
        y_r_m = self.get_layer_encoding(layer_id)
        y_r_m_1 = self.get_layer_encoding(layer_id-1)
        y_r_m_1.append(-1)
        
        self.propagate(different_class_shortest_distance_pair[1])
        y_s_m = self.get_layer_encoding(layer_id)
        y_s_m_1 = self.get_layer_encoding(layer_id-1)
        y_s_m_1.append(-1) 
        
        for node_id in range(0,len(self.layers[layer_id])):
            for edge_id in range(0,len(self.layers[layer_id][node_id].in_edge)):
                self.layers[layer_id][node_id].in_edge[edge_id].weight -= (self.eta_att*(+((y_p_m[node_id]-y_q_m[node_id])*(y_p_m[node_id]-y_p_m[node_id]*y_p_m[node_id])*(y_p_m_1[edge_id])) -((y_p_m[node_id]-y_q_m[node_id])*(y_q_m[node_id]-y_q_m[node_id]*y_q_m[node_id])*(y_q_m_1[edge_id]))) + self.eta_rep *(-((y_r_m[node_id]-y_s_m[node_id])*(y_r_m[node_id]-y_r_m[node_id]*y_r_m[node_id])*(y_r_m_1[edge_id]))+((y_r_m[node_id]-y_s_m[node_id])*(y_s_m[node_id]-y_s_m[node_id]*y_s_m[node_id])*(y_s_m_1[edge_id]))))
            self.layers[layer_id][node_id].bias -= (self.eta_att*(((y_p_m[node_id]-y_q_m[node_id])*(y_p_m[node_id]-y_p_m[node_id]*y_p_m[node_id])*(-1)) -((y_p_m[node_id]-y_q_m[node_id])*(y_q_m[node_id]-y_q_m[node_id]*y_q_m[node_id])*(-1))) + self.eta_rep *(-((y_r_m[node_id]-y_s_m[node_id])*(y_r_m[node_id]-y_r_m[node_id]*y_r_m[node_id])*(-1))+((y_r_m[node_id]-y_s_m[node_id])*(y_s_m[node_id]-y_s_m[node_id]*y_s_m[node_id])*(-1))))

    def get_same_class_longest_distance_pair(self,trainning_data,layer_id):

        trainning_data_including_y = []
        for single_trainning_data in trainning_data:
            single_trainning_data_including_y = [single_trainning_data[0]]
            self.propagate(single_trainning_data[0])
            single_trainning_data_including_y.append(self.get_layer_encoding(layer_id))
            single_trainning_data_including_y.append(single_trainning_data[1])
            trainning_data_including_y.append(single_trainning_data_including_y)

        max_distance_pair = [[],[]]
        max_distance = -1
        for single_trainning_data_including_y in  trainning_data_including_y:
            for single_trainning_data_including_y_compare in  trainning_data_including_y:
                if single_trainning_data_including_y != single_trainning_data_including_y_compare and single_trainning_data_including_y[2] == single_trainning_data_including_y_compare[2]:
                    distance = 0
                    for i in range(0,len(single_trainning_data_including_y[1])):
                        distance += abs(single_trainning_data_including_y[1][i] - single_trainning_data_including_y_compare[1][i])*abs(single_trainning_data_including_y[1][i] - single_trainning_data_including_y_compare[1][i])
                    if distance > max_distance:
                        max_distance = distance
                        max_distance_pair = [single_trainning_data_including_y[0],single_trainning_data_including_y_compare[0]]

        print(max_distance, end="\t")
        return max_distance_pair

    def get_different_class_shortest_distance_pair(self,trainning_data,layer_id):

        trainning_data_including_y = []
        for single_trainning_data in trainning_data:
            single_trainning_data_including_y = [single_trainning_data[0]]
            self.propagate(single_trainning_data[0])
            single_trainning_data_including_y.append(self.get_layer_encoding(layer_id))
            single_trainning_data_including_y.append(single_trainning_data[1])
            trainning_data_including_y.append(single_trainning_data_including_y)

        min_distance_pair = [[],[]]
        min_distance = 9999
        for single_trainning_data_including_y in  trainning_data_including_y:
            for single_trainning_data_including_y_compare in  trainning_data_including_y:
                if single_trainning_data_including_y != single_trainning_data_including_y_compare and single_trainning_data_including_y[2] != single_trainning_data_including_y_compare[2]:
                    distance = 0
                    for i in range(0,len(single_trainning_data_including_y[1])):
                        distance += abs(single_trainning_data_including_y[1][i] - single_trainning_data_including_y_compare[1][i])*abs(single_trainning_data_including_y[1][i] - single_trainning_data_including_y_compare[1][i])
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_pair = [single_trainning_data_including_y[0],single_trainning_data_including_y_compare[0]]

        print(min_distance)
        return min_distance_pair

    def get_layer_encoding(self,layer_id):
        layer_encoding = []
        for node in self.layers[layer_id]:
            layer_encoding.append(node.value)
        return layer_encoding

    def propagate(self,single_trainning_data_x):

        for input_layer_node_id in range(0,len(self.layers[0])):
            self.layers[0][input_layer_node_id].value = -1+2*sigmoid(single_trainning_data_x[input_layer_node_id])

        for hiddenlayer in self.layers[1:]:
            for hiddenlayer_node in hiddenlayer:
                tmp_sum = 0
                for edge in hiddenlayer_node.in_edge:
                    tmp_sum += edge.from_node.value*edge.weight
                tmp_sum += hiddenlayer_node.bias
                hiddenlayer_node.value = -1+2*sigmoid(tmp_sum)

if __name__ == '__main__':

    #build demo_som_perceptron
    demo_som_perceptron = som_perceptron([2,5,5,5,5,5])
    demo_som_perceptron.init_random_weights()
    
    '''  
    build demo_trainning_data from demo files,you can use anyway to build the trainning data,just follow the format shown bellow

    example:
    demo_trainning_data = [
        [[0.489735712483345, 0.970473278829661], 1],
        [[-0.607590162537496, 0.658111817557601], 1],
        [[0.974975150378706, -0.811022872158843], 0],
        [[0.0431793529961098, -0.333143627078267], 2]
    ]
    
    take the first data [[0.489735712483345, 0.970473278829661], 1] in the demo_trainning_data set as example
    [0.489735712483345, 0.970473278829661] is the data position which could been set in any dimension
    1 is the class of this data

    '''
    demo_trainning_data = []
    demo_trainning_data_file = open("hw2pt.dat","r",encoding = "utf-8")
    demo_trainning_data_class_file = open("hw2class.dat","r",encoding = "utf-8")
    for line in demo_trainning_data_file:
        demo_trainning_data_x = []
        demo_trainning_data_x.append(float(line[:-1].split("	")[0]))
        demo_trainning_data_x.append(float(line[:-1].split("	")[1]))
        demo_trainning_data.append([demo_trainning_data_x])
    for line in demo_trainning_data_class_file:
        for single_demo_trainning_data_class_id in range(0,len(line[:-1].split("\t"))):
            demo_trainning_data[single_demo_trainning_data_class_id].append(int(line[:-1].split("\t")[single_demo_trainning_data_class_id]))
        break
    demo_trainning_data_file.close()
    demo_trainning_data_class_file.close()
    
    #trainning demo_som_perceptron with demo_trainning_data
    demo_som_perceptron.train(demo_trainning_data,0.00001,0.1,5000)










