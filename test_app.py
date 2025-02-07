import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Checkbutton, IntVar, Scale, HORIZONTAL, OptionMenu, StringVar
from torch import nn
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.ToPILImage()
])

def add_noise(image_tensor, noise_level):
    noise = torch.randn_like(image_tensor) * noise_level
    noisy_image = image_tensor + noise
    return noisy_image

def classify_image(image_path, model, class_names, add_noise_flag, noise_level):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    if add_noise_flag:
        input_tensor = add_noise(input_tensor, noise_level)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()], input_tensor if add_noise_flag else None

class_names = ["Cat", "Dog"]

models_paths = {
    "Original Dataset": "models/bs100_lr0.001_epochs10.pth",
    "Noisy Dataset": "models/bs100_lr0.001_epochs10_noisy.pth",
    "Combined Dataset": "models/bs100_lr0.001_epochs20_combined.pth"
}

root = tk.Tk()
root.title("Image Classifier")

selected_model = StringVar(value="Original Dataset")

model = load_model(models_paths[selected_model.get()], num_classes=len(class_names))

def select_and_classify_multiple():
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        for widget in results_frame.winfo_children():
            widget.destroy()

        num_columns = 5
        row, col = 0, 0

        for file_path in file_paths:
            noise_level = noise_slider.get() / 100.0
            predicted_class, noisy_tensor = classify_image(file_path, model, class_names, noise_var.get(), noise_level)

            if noise_var.get() and noisy_tensor is not None:
                noisy_image = inverse_transform(noisy_tensor.squeeze(0).cpu())
                img = noisy_image.resize((100, 100))
            else:
                img = Image.open(file_path).resize((100, 100))

            img_tk = ImageTk.PhotoImage(img)
            img_label = Label(results_frame, image=img_tk)
            img_label.image = img_tk
            img_label.grid(row=row, column=col, padx=10, pady=10)

            text_label = Label(results_frame, text=f"{predicted_class}", font=("Helvetica", 10))
            text_label.grid(row=row + 1, column=col, padx=10, pady=10)

            col += 1
            if col >= num_columns:
                col = 0
                row += 2

def update_model(*args):
    global model
    model = load_model(models_paths[selected_model.get()], num_classes=len(class_names))

label_model_select = Label(root, text="Select Model:")
label_model_select.pack()

model_select = OptionMenu(root, selected_model, *models_paths.keys(), command=update_model)
model_select.pack()

noise_var = IntVar()
checkbox_noise = Checkbutton(root, text="Add noise to images", variable=noise_var)
checkbox_noise.pack()

noise_slider_label = Label(root, text="Noise Level:")
noise_slider_label.pack()
noise_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
noise_slider.set(50)
noise_slider.pack()

button_select = Button(root, text="Select Images", command=select_and_classify_multiple)
button_select.pack()

results_frame = Frame(root)
results_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
