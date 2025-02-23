import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size=5, output_size=1, learning_rate=0.1, epochs=50000):
        """
        Initialize the Multi-Layer Perceptron (MLP) with random weights and biases.
        """
        self.W1 = np.random.randn(hidden_size, input_size)  # Weights for input to hidden layer
        self.b1 = np.zeros((hidden_size, 1))  # Biases for hidden layer
        self.W2 = np.random.randn(output_size, hidden_size)  # Weights for hidden to output layer
        self.b2 = np.zeros((output_size, 1))  # Biases for output layer
        self.lr = learning_rate  # Learning rate for weight updates
        self.epochs = epochs  # Number of training iterations
    
    def step_function(self, x):
        """Step activation function: Returns 1 if x >= 0, else 0."""
        return np.where(x >= 0, 1, 0)

    def forward(self, X):
        """Perform forward propagation."""
        self.z1 = np.dot(self.W1, X) + self.b1  # Compute input to hidden layer
        self.a1 = self.step_function(self.z1)   # Apply step activation function
        self.z2 = np.dot(self.W2, self.a1) + self.b2  # Compute input to output layer
        self.a2 = self.step_function(self.z2)   # Apply step activation function
        return self.a2

    def backward(self, X, y, output):
        """Perform backward propagation using perceptron weight update rule."""
        error = y - output  # Compute error between expected and actual output
        
        # Update weights and biases using perceptron learning rule
        self.W2 += self.lr * np.dot(error, self.a1.T)  # Adjust weights of hidden to output layer
        self.b2 += self.lr * np.sum(error, axis=1, keepdims=True)  # Adjust biases for output layer
        self.W1 += self.lr * np.dot(np.dot(self.W2.T, error), X.T)  # Adjust weights of input to hidden layer
        self.b1 += self.lr * np.sum(np.dot(self.W2.T, error), axis=1, keepdims=True)  # Adjust biases for hidden layer
    
    def train(self, X, y):
        """Train the MLP using the given dataset."""
        for epoch in range(self.epochs):
            output = self.forward(X)  # Perform forward pass
            self.backward(X, y, output)  # Perform backward pass and update weights
            loss = np.mean((y - output) ** 2)  # Compute mean squared error as loss
            acc = self.accuracy(X, y)  # Compute accuracy
            
            # Print training progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
    
    def predict(self, X):
        """Make predictions using trained model."""
        return self.forward(X)
    
    def accuracy(self, X, y):
        """Calculate the accuracy of the model."""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)  # Count correct predictions
        return correct / y.shape[1] * 100  # Compute accuracy percentage

# Define XOR dataset
X = np.array([[0, 0, 1, 1],  # First input feature
              [0, 1, 0, 1]])  # Second input feature
Y = np.array([[0, 1, 1, 0]])  # Expected XOR outputs

# Initialize and train the MLP
mlp = MLP(input_size=2, hidden_size=5, output_size=1)
mlp.train(X, Y)

# Test the trained MLP
print("\nTesting Trained MLP:")
for i in range(X.shape[1]):
    x_sample = X[:, i].reshape(-1, 1)  # Extract single input sample
    output = mlp.predict(x_sample)  # Get model prediction
    print(f"Input: {X[:, i]}, Predicted Output: {output.flatten()[0]}")