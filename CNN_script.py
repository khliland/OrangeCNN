# Don't touch!
import httpimport
with httpimport.remote_repo(['Simple_CNN'], 'https://raw.githubusercontent.com/khliland/OrangeCNN/master'):
    from Simple_CNN import *

######################
## Apply simple CNN ##
######################
        
classify      = True
epochs        = 100
batch_size    = 64
learning_rate = 0.0001

(X_train, Y_train, X_test, Y_test) = dataPrep(in_data)
history = CNN(classify, batch_size=batch_size, epochs=epochs, lr=learning_rate)

