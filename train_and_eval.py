from AlexNet_class import AlexNet
import matplotlib.pyplot as plt

# Initialize AlexNet-like model
alexnet_model = AlexNet()

# Train autoencoder model
alexnet_model.train_model()

# Predict on testing dataset
categorical_loss, pred_labels, true_labels = alexnet_model.eval_model()

# Store losses to plot learning curves
val_loss = alexnet_model.training_history.history["val_loss"]

# Plot learning curve vs iterations
plt.figure(figsize=(12, 8))
plt.plot(val_loss, label=f"Learning curve vs epochs)")
plt.title("Learning Curves vs Epochs (Validation Dataset)")
plt.xlabel("Epochs")
plt.ylabel("Categorical Cross Entropy")
plt.show()

# Generate confusion matrix
confusion_matrix = alexnet_model.generate_confusion_matrix(pred_labels, true_labels)

# Calculate performance stats
