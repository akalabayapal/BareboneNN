# BareboneNN

An educational NN (Neural Network) libary built from pure math.

## Usage


    from main import BareboneNN
    from layers import *
    from loss_function import *


    model = BareboneNN(
        LinearLayer(num_neurons,learning_rate),
        ReluLayer(),
        LinearLayer(1,learning_rate),
        SigmoidLayer()
    )

    # Setup a loass function
    model.loss_function = CrossEntropyBinaryLoss


    model.fit(X,Y,epoch)


    # get final error
    print(model.error)

    # get error progression over epoches
    print(model.errors)

    # predict
    print(model.predict(
        prediction_array
    ))

### Note: Project is under active delelopment

    
