import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

def generate_unique_name(base_name, folder):
    i = 1
    unique_name = base_name
    while os.path.exists(os.path.join(folder, unique_name)):
        unique_name = f"{base_name}_{i}"
        i += 1
    return unique_name

def create_model(num_classes):
    resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    return resnet_model.to(device)

def train_model(model, loader, optimizer, criterion, writer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        writer.add_scalar("Train/Loss", loss.item(), epoch * len(loader) + i)
    return running_loss / len(loader)

def evaluate_model(model, loader, criterion, writer=None, epoch=None, phase="Validation"):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = sum([y_t == y_p for y_t, y_p in zip(y_true, y_pred)]) / len(y_true)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    if writer and epoch is not None:
        writer.add_scalar(f"{phase}/Loss", running_loss / len(loader), epoch)
        writer.add_scalar(f"{phase}/Accuracy", accuracy, epoch)
        writer.add_scalar(f"{phase}/Precision", precision, epoch)
        writer.add_scalar(f"{phase}/Recall", recall, epoch)
        writer.add_scalar(f"{phase}/F1-Score", f1, epoch)

    return running_loss / len(loader), accuracy, precision, recall, f1

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
input_size = (224, 224)
batch_size = 100
epochs = 10
learning_rate = 1e-3
# Control dataset selection
use_noise_dataset = False  # Set True for noisy dataset
use_combined_dataset = True  # Set True for combined dataset

if use_noise_dataset:
    train_path = "C:/Users/coval/Desktop/mini - project/dataset_noisy/train"
elif use_combined_dataset:
    train_path = "C:/Users/coval/Desktop/mini - project/dataset_resized_combined/train"
else:
    train_path = "C:/Users/coval/Desktop/mini - project/dataset/train"

test_path = "C:/Users/coval/Desktop/mini - project/dataset/test"

# Data transformations
transform_train = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
train_dataset = ImageFolder(train_path, transform=transform_train)
test_dataset = ImageFolder(test_path, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, and optimizer
num_classes = len(train_dataset.classes)
model = create_model(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Initialize TensorBoard writer with a unique name
base_folder = "runs"
base_name = f"bs{batch_size}_lr{learning_rate}_epochs{epochs}{'_noisy' if use_noise_dataset else ('_combined' if use_combined_dataset else '')}"
unique_name = generate_unique_name(base_name, base_folder)
writer = SummaryWriter(os.path.join(base_folder, unique_name))

# Ensure the model save directory exists
model_save_folder = "models"
os.makedirs(model_save_folder, exist_ok=True)
model_save_path = os.path.join(model_save_folder, f"{unique_name}.pth")

# Training loop
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, writer, epoch)
    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, test_loader, criterion, writer, epoch, "Validation")
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
          f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)

# Close the TensorBoard writer
writer.close()
