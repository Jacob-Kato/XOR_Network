import network
Network = network.XORNetwork()

learning_rate = 0.01
 
dataset = [[[0,0], 0],
           [[1,0], 1],
           [[0,1], 1],
           [[1,1], 0]
]


for epoch in range(100000):     

    for x, y in dataset:
        prediction = Network.forward(x)
        total_loss = 0
        total_loss += (prediction - y)**2
        Network.backward(x,y,learning_rate)
            

        if epoch % 2500 == 0:
            print(f"""epoch is {epoch}: \n when x is {x} \n your prediction is {round(prediction)}
true prediction is : {y} \n your loss is {total_loss}""")
            print("\n")

        
