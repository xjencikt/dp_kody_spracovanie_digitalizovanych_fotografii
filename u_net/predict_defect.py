import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from codes.u_net.my_Unet import UNet

def main():
    # Define U-Net model
    model = UNet(n_classes=1)
    model.load_state_dict(torch.load("../final_scratch_model.pth", map_location=torch.device("cpu")))
    model.eval()

    threshold = 0.25
    image_path = "E79222E2X120209_05112024_095325_001574.jpg"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # Display original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Originalna fotka, threshold: " + str(threshold))
    plt.axis("off")

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process output
    output_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (output_mask > threshold).astype("uint8") * 255  # Convert to 0–255

    print("Tvar vystupu modelu:", output.shape)
    print("Minimalna hodnota tensoru:", output.min().item())
    print("Maximalna hodnota tensoru:", output.max().item())

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Optionally apply closing to fill small holes:
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Resize cleaned mask to match original image size
    cleaned_mask_resized = cv2.resize(cleaned_mask, image.size)

    # Show cleaned mask
    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_mask_resized, cmap="jet")
    plt.colorbar()
    plt.show()

    # Save the cleaned mask
    mask_saved = "predicted_defect_img.png"
    Image.fromarray(cleaned_mask_resized).save(mask_saved)
    print(f"Maska ulozena: {mask_saved}")

if __name__ == "__main__":
    main()
