from AlexNet_class import AlexNet, AlexNet2, AlexNet3
import matplotlib.pyplot as plt

# Initialize a AlexNet-like model
alexnet_model = AlexNet3()

# Train autoencoder model
alexnet_model.train_model()

# Predict on testing dataset
categorical_loss, test_accuracy, pred_labels, true_labels = alexnet_model.eval_model()
print(f"Accuracy on testing dataset is {test_accuracy}")

# Store losses to plot learning curves
train_loss = alexnet_model.training_history.history["loss"]
val_loss = alexnet_model.training_history.history["val_loss"]

# Plot learning curve vs iterations
plt.figure(figsize=(12, 8))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Learning Curves vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Categorical Cross Entropy")
plt.legend()
plt.show()

# Plot accuracy curve
training_accuracy = alexnet_model.training_history.history["accuracy"]
val_accuracy = alexnet_model.training_history.history["val_accuracy"]

plt.figure(figsize=(10, 5))
plt.plot(training_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Accuracies vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Generate confusion matrix
confusion_matrix = alexnet_model.generate_confusion_matrix(pred_labels, true_labels)
print("Confusion matrix of predictions:")
print(confusion_matrix)
