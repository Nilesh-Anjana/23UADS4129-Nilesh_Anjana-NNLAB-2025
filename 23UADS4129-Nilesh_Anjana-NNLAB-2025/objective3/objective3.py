import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Disable eager execution for using sessions (TensorFlow 2.x compatibility with TF1.x-style sessions)
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset (dataset of handwritten digits 0-9)
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# Function to preprocess dataset (normalize images, flatten, and one-hot encode labels)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to range [0,1]
    image = tf.reshape(image, [-1])  # Flatten 28x28 image into a 1D vector (size 784)
    label = tf.one_hot(label, depth=10)  # Convert label (0-9) into a one-hot encoded vector of size 10
    return image, label

# Apply preprocessing function to dataset and create mini-batches of 32 samples
ds_train = ds_train.map(preprocess).batch(32)
ds_test = ds_test.map(preprocess).batch(32)

# Create iterators for datasets
dataset_iterator = tf.compat.v1.data.make_initializable_iterator(ds_train)
next_batch = dataset_iterator.get_next()

dataset_test_iterator = tf.compat.v1.data.make_initializable_iterator(ds_test)
next_test_batch = dataset_test_iterator.get_next()

# Define placeholders for inputs and labels
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])  # Input images
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])  # True labels

# Define model parameters
input_size = 784  # 28x28 pixels flattened
hidden_size = 128  # Hidden layer neurons
output_size = 10  # Output classes
learning_rate = 0.001  # Learning rate
epochs = 5  # Training epochs

# Xavier initialization
initializer = tf.compat.v1.initializers.glorot_uniform()

# Initialize weights and biases
W1 = tf.Variable(initializer([input_size, hidden_size]))
b1 = tf.Variable(tf.zeros([hidden_size]))
W2 = tf.Variable(initializer([hidden_size, output_size]))
b2 = tf.Variable(tf.zeros([output_size]))

# Forward propagation
hidden_input = tf.matmul(X, W1) + b1
hidden_output = tf.nn.sigmoid(hidden_input)
final_input = tf.matmul(hidden_output, W2) + b2
final_output = tf.nn.softmax(final_input)

# Define loss function
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_input))

# Define optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_fn)

# Define accuracy calculation
correct_prediction = tf.equal(tf.argmax(final_output, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables
init = tf.compat.v1.global_variables_initializer()

# Train the model
with tf.compat.v1.Session() as sess:
    sess.run(init)  # Initialize all variables
    
    for epoch in range(epochs):  # Loop over epochs
        sess.run(dataset_iterator.initializer)  # Reinitialize dataset iterator
        total_loss = 0  # Track total loss
        num_batches = 0  # Track number of batches
        
        while True:
            try:
                X_batch_np, Y_batch_np = sess.run(next_batch)  # Fetch next batch
                _, batch_loss = sess.run([optimizer, loss_fn], feed_dict={X: X_batch_np, Y: Y_batch_np})
                total_loss += batch_loss  # Accumulate loss
                num_batches += 1  # Increment batch count
            except tf.errors.OutOfRangeError:
                break  # Break when dataset is exhausted
        
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No data processed!")
    
    # Evaluate model accuracy
    sess.run(dataset_test_iterator.initializer)  # Initialize test dataset iterator
    total_correct = 0  # Correct predictions
    total_samples = 0  # Total samples
    
    while True:
        try:
            X_batch_np, Y_batch_np = sess.run(next_test_batch)  # Fetch next batch
            acc = sess.run(accuracy, feed_dict={X: X_batch_np, Y: Y_batch_np})
            total_correct += acc * len(X_batch_np)
            total_samples += len(X_batch_np)
        except tf.errors.OutOfRangeError:
            break  # Break when dataset is exhausted
    
    print(f"Test Accuracy: {100 * (total_correct / total_samples):.2f}%")
