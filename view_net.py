import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define funcția pentru a desena o arhitectură în stil schematic
def plot_resnet50_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Etichete pentru fiecare etapă
    stages = [
        {"name": "Input Image", "size": "224x224", "color": "green"},
        {"name": "7x7 Conv, 64/2", "size": "112x112", "color": "red"},
        {"name": "3x3 MaxPool", "size": "56x56", "color": "blue"},
        {"name": "Stage 1: 1x1, 3x3, 1x1 Conv (64, 64, 256)", "size": "56x56", "color": "red"},
        {"name": "Stage 2: 1x1, 3x3, 1x1 Conv (128, 128, 512)", "size": "28x28", "color": "red"},
        {"name": "Stage 3: 1x1, 3x3, 1x1 Conv (256, 256, 1024)", "size": "14x14", "color": "red"},
        {"name": "Stage 4: 1x1, 3x3, 1x1 Conv (512, 512, 2048)", "size": "7x7", "color": "red"},
        {"name": "2x2 Global Pool", "size": "1x1", "color": "blue"},
        {"name": "FC 1000", "size": "1x1", "color": "red"},
    ]

    # Desenează etapele pe grafic
    x_offset = 0
    for stage in stages:
        # Adaugă un dreptunghi pentru fiecare etapă
        rect = Rectangle((x_offset, 0.5), width=2, height=1, color=stage["color"], alpha=0.7)
        ax.add_patch(rect)
        # Adaugă eticheta etapei
        ax.text(x_offset + 1, 1.1, stage["name"], ha="center", va="bottom", fontsize=9)
        ax.text(x_offset + 1, 0.4, stage["size"], ha="center", va="top", fontsize=8)
        x_offset += 2.5

    # Setează limitele graficului
    ax.set_xlim(0, x_offset)
    ax.set_ylim(0, 2)
    plt.show()

# Rulează funcția pentru a vizualiza arhitectura
plot_resnet50_architecture()
