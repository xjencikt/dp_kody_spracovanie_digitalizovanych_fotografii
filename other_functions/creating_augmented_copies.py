from codes.other_functions.my_functions import *

def main():
    input_dir = '../DP2/generated_images_and_masks_2.1/input'
    output_dir = '../DP2/generated_images_and_masks_2.1/extracted_objects'

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.5),
    ])

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
            image_path = os.path.join(input_dir, filename)
            augment_image(transform, image_path, output_dir)


if __name__ == "__main__":
    main()