# Don't touch! ------->
import httpimport
with httpimport.remote_repo('https://raw.githubusercontent.com/khliland/OrangeCNN/master'):
    from Simple_CNN import *
# <------- Don't touch!


######################
## Apply simple CNN ##
######################
        
classify      = True
epochs        = 100
batch_size    = 64
learning_rate = 0.0001


# Don't touch! ------->
(X_train, Y_train, X_test, Y_test) = dataPrep(in_data)
history = CNN(classify, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, batch_size=batch_size, epochs=epochs, lr=learning_rate)
out_data = Table.from_numpy(None,np.hstack([np.array(history.history['accuracy'])[:,np.newaxis], np.arange(1,epochs + 1,1)[:,np.newaxis]]))
# <------- Don't touch!
