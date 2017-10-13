# OO Neural Network

Almost all of the neural network implementation I found in public domain is matrix-based.But Object-based implementation with real graph composed by nodes and edges is much more interesting to me.So I implement one for fun.

***
Back Propagation Usage:

Build a network
network = network(input layer size,layer1 size,layer2 size,...,layern size,output layer size) 
* n = network(2,4,3,1)


Initialize all the weights of the network
network.init_random_weights()
* n.init_random_weights()


Input trainning data example

trainning_data = [

[[0,0], [0]],

[[0,1], [1]],

[[1,0], [1]],

[[1,1], [0]]]


Start trainning
network.back_propagation_train(trainning data,iteration)
* n.back_propagation_train(trainning_data,3000)


After trainning , predict the testing data
network.back_propagation_predict(testing_data) 
return testing data result, the format is the same as the testing data
* predict_data = n.back_propagation_predict(trainning_data)

Result

predict_data =[

[[0, 0], [0.04146216018413632]],

[[0, 1], [0.9646378727416303]],

[[1, 0], [0.9640453475598098]],

[[1, 1], [0.03672575912926116]]

***
Som Perceptron Usage:

Implementation of *"Cheng-Yuan Liou and Wei-Chen Cheng (2011), Forced Accretion and Assimilation Based on Self-organizing Neural Network, Self Organizing Maps - Applications and Novel Algorithm Design,Chapter 35 in Book edited by: Josphat Igadwa Mwasiagi, page 683-702, ISBN: 978-953-307-546-4,Publisher: InTech, Publishing date: January 2011"*.

Build a network with any number of layers with specific sizes
network = network([input layer size,layer1 size,layer2 size,...]) 
* n = som_perceptron([2,5,5,5,5,5])


Initialize all the weights of the network
network.init_random_weights()
* n.init_random_weights()

trainning_data = [

        [[0.489735712483345, 0.970473278829661], 1],

        [[-0.607590162537496, 0.658111817557601], 1],

        [[0.974975150378706, -0.811022872158843], 0],

        [[0.0431793529961098, -0.333143627078267], 2]

    ]
    
Take the first data [[0.489735712483345, 0.970473278829661], 1] in the trainning_data set as example[0.489735712483345, 0.970473278829661] is the data position which could been set in any dimension 1 is the class of this data


Start trainning
network.train(trainning_data,eta_att,eta_rep,epoch)
* n.train(trainning_data,0.00001,0.1,5000)
