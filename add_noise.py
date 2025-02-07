import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def add_noise(image_tensor):
    noise = torch.randn_like(image_tensor) * 0.1
    noisy_image = image_tensor + noise
    return noisy_image

transform = transforms.ToTensor()

inverse_transform = transforms.ToPILImage()

def save_noisy_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    dataset = ImageFolder(input_folder, transform=transform)

    for i, (img_tensor, label) in enumerate(dataset):
        noisy_img_tensor = add_noise(img_tensor)
        noisy_img = inverse_transform(noisy_img_tensor)

        class_folder = os.path.join(output_folder, dataset.classes[label])
        os.makedirs(class_folder, exist_ok=True)

        noisy_img.save(os.path.join(class_folder, f"image_{i}.png"))

input_folder = "C:/Users/coval/Desktop/mini - project/dataset/train"
output_folder = "C:/Users/coval/Desktop/mini - project/dataset_noisy/train"

save_noisy_images(input_folder, output_folder)
print(f"Imaginile cu zgomot au fost salvate în {output_folder}")
