<h1>Object 7</h1>
<h4>  WAP to retrain a pretrained imagenet model to classify a medical image dataset.  </h4>
<hr>
<h3>Description of the Model : </h3>
<p>This project uses transfer learning with the VGG16 model to classify CT scan images into two categories: COVID and Non-COVID. The dataset is organized into two folders and is split into training and validation sets using an 80-20 ratio. Data augmentation techniques like resizing, flipping, and rotation are applied to improve the model's generalization. The original VGG16 layers are mostly frozen to retain learned features, with only the last convolutional block (from layer 24 to 30) fine-tuned. A custom classifier is added to the model to perform binary classification using sigmoid activation.

The model is trained using the Binary Cross Entropy Loss function and optimized with the Adam optimizer. It runs for 20 epochs, tracking both training and validation accuracy and loss. After training, the model is saved for future use, and its performance is visualized using plots. The final accuracy on the validation set is also reported. This approach provides a reliable way to apply deep learning to medical image classification, even with a relatively small dataset.</p>

<hr>

<h3>Description about the code</h3>

<b>1. Libraries and Hyperparameters : </b>The script begins by importing essential libraries such as torch, torchvision, and matplotlib. These are used for deep learning, image processing, and plotting. Key hyperparameters like image size, batch size, number of epochs, and learning rate are defined, along with selecting the computation device (GPU if available, else CPU).

<b>2. Data Transformations : </b>To prepare the dataset, two types of image transformations are used. The training data is augmented with resizing, random horizontal flips, and rotations to improve model generalization. Both training and validation data are normalized to match the input requirements of the pre-trained VGG16 model (ImageNet mean and std).

<b>3. Dataset Loading and Splitting : </b>The dataset is loaded from a folder structure using ImageFolder, where each subfolder represents a class (CT_COVID and CT_NonCOVID). The full dataset is split into training and validation sets using an 80-20 split. DataLoader is used to feed images in batches for both training and validation with shuffling enabled for training data.

<b>4. Model Setup: VGG16 : </b>A pre-trained VGG16 model is loaded to leverage features learned from ImageNet. All layers are initially frozen to preserve learned features, except the last convolutional block which is unfrozen for fine-tuning. This allows the model to slightly adapt to the new dataset while retaining most of the pretrained knowledge.

<b> 5. Custom Classifier Design : </b>The original classifier of VGG16 is replaced with a custom fully connected network tailored for binary classification. It consists of multiple Linear, ReLU, and Dropout layers, ending in a single neuron with a Sigmoid activation. This outputs a probability between 0 and 1, indicating the class (COVID or Non-COVID).

<b> 6. Loss Function and Optimizer : </b>The Binary Cross Entropy Loss (BCELoss) is used since the task is binary classification. The Adam optimizer is chosen for its efficiency and ability to adapt the learning rate. Only the parameters that require gradients (i.e., the unfrozen layers and custom classifier) are passed to the optimizer.

<b> 7. Training Loop : </b>The model is trained over 20 epochs. For each batch, it performs a forward pass, computes loss, backpropagates the error, and updates the weights. Training loss and accuracy are tracked after every epoch. The model is then evaluated on the validation set to track performance on unseen data.

<b> 8. Validation and Metric Tracking : </b>
During validation, the model is set to evaluation mode to prevent dropout and batch norm updates. Accuracy and loss are computed without updating the weights. This helps monitor if the model is overfitting or improving generalization. All metrics are stored for visualization.

<b>9. Saving and Plotting Results : </b>After training is complete, the model is saved to a file named covid_classifier_vgg16.pt for future use. Accuracy and loss trends for both training and validation sets are plotted using matplotlib, helping visualize how the model performed across epochs.

<b> 10. Final Evaluation : </b>Lastly, the model is evaluated one final time on the validation set to report its test accuracy. This provides a clear measure of how well the trained model can generalize to new CT scan images. The result is printed to the console for quick assessment.

<hr>

<h3>My Comments</h3>
<ol><li>The architecture referred from the OpenAI ChatGPT was giving around 72% test accuracy.</li>
<li>On modifying the classifier portion of the architecture there was not much improvement in the model accuracy (~1%).</li>
<li>However the significant improvement in the accuracy is observed when the output from the last convolutional block (from layer 24 to 30) is unfreezed and included in the training process.</li>  
<li>The prospect of changing the dropout rate and exploring batch normalisation for improving the test accuracy has not been extensively explored and there might be a possibility of finding even a better architecture by thinking in this direction. 
</ol>









