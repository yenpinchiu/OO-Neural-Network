#Objective Neural Network Som Perceptron

The objective oriented implementation of *"Cheng-Yuan Liou and Wei-Chen Cheng (2011), Forced Accretion and Assimilation Based on Self-organizing Neural Network, Self Organizing Maps - Applications and Novel Algorithm Design,Chapter 35 in Book edited by: Josphat Igadwa Mwasiagi, page 683-702, ISBN: 978-953-307-546-4,Publisher: InTech, Publishing date: January 2011"*.

usage:

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
