######################
## Apply simple CNN ##
######################
        
classify      = True
epochs        = 100
batch_size    = 64
learning_rate = 0.0001

history = CNN(classify, batch_size=batch_size, epochs=epochs, lr=learning_rate)

