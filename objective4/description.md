• The code starts by loading the MNIST dataset, normalizing the images (scaling pixel values between 0 and 1), and reshaping them into 1D vectors. Labels are also converted into one-hot encoding for easy processing.  

• Since the code uses TensorFlow 1.x, eager execution is disabled to ensure compatibility with older TensorFlow functions.  

• Placeholders (X for input images and Y for labels) are defined so that data can be fed in batches during training.  

• Model parameters like weights and biases are initialized randomly for both hidden and output layers.  

• Different models are tested with hidden layers of 256, 128, and 64 neurons, using activation functions like ReLU, Sigmoid, and Tanh.  

• The forward propagation step computes activations using matrix multiplication (tf.matmul), and the chosen activation function is applied. The output layer produces logits without softmax.  

• The model uses softmax cross-entropy loss to measure errors, and it is optimized using Gradient Descent to adjust weights and improve accuracy.  

• Training runs for 50 epochs with small batches of 10 images at a time. After each epoch, the loss and accuracy are recorded.  

• Once training is done, the model is tested on unseen data, and key results like test accuracy and confusion matrices are generated.  

• Finally, all results—accuracy, loss trends, and confusion matrices—are stored in a DataFrame, and graphs are plotted to visualize performance.  
