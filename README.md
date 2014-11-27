#objective neural network back propagation

Almost all of the neural network back propagation implementation I found in public domain is matrix-based.I think the object-based implementation with real graph composed by nodes and edges is much more intuitive to me.

usage:

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
