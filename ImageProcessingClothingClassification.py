# Author:       Ethan Pattison
# FSU Course:   SENG 609
# Professor:    Dr Abusharkh
# Assingment:   Assignment 9: Task 2
# Date:         10/11/2022



# This guide trains a neural network model to classify images of clothing, like sneakers and shirts.
# It's okay if you don't understand all the details;
# This is a fast-paced overview of a complete TensorFlow program with the details explained as you go.

# This guide uses tf.keras, a high-level API to build and train models in TensorFlow.

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Check the version of tensor flow
print(tf.__version__)



# Import the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Import the arrays for the training and testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



# Store the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Get the number of images in the training set / and pixels size of image
train_images.shape


# Check the count of labels
len(train_labels)


# In[6]:


# Labals are integers between 1 and 9
train_labels


# In[7]:


# Get the number of images in the testing set / and pixels size of image
test_images.shape



# Check the count of labels
len(test_labels)


# Inspect the first image in the training set for range of pixels
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



# scale the values to the range of 0-1 before using neural network
train_images = train_images / 255.0

test_images = test_images / 255.0


# Verify that the data is in the correct format
# Display the first 25 images from the training set

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



# Build the model / Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])



# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# Feed the model
model.fit(train_images, train_labels, epochs=10)




# Evaluate accuracy of the NN
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)



# Attach a softmax layer to convert the model's linear outputs—logits—to probabilities
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])



# predict the label for each image in the testing set
predictions = probability_model.predict(test_images)



# look at the first prediction:
predictions[0]



# A prediction is an array of 10 numbers.
# They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
np.argmax(predictions[0])




# Examining the test label shows that this classification is correct:
test_labels[0]



# Graph this to look at the full set of 10 class predictions

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# Verify predictions - Correct prediction labels are blue and incorrect prediction labels are red
# The number gives the percentage (out of 100) for the predicted label
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()



# Verify predictions
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()



# Let's plot several images with their predictions. Note that the model can be wrong even when very confident
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# Use trained model
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)


# Even though you're using a single image, you need to add it to a list
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# Now predict the correct label for this image
predictions_single = probability_model.predict(img)

print(predictions_single)


# Grab the predictions for the only image in the batch
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()



# Verify prediction of label 2 as expected
np.argmax(predictions_single[0])
