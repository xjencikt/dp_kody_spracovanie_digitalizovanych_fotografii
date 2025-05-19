from codes.other_functions.my_functions import *

def main():
    input_image_dir = "../DP2/generated_images_and_masks_2.1/input/"
    mask_dir = "../DP2/generated_images_and_masks_2.1/mask/"
    output_dir = "../DP2/generated_images_and_masks_2.1/extracted_objects/"

    image_files = []
    get_files(image_files, input_image_dir)

    for image_file in image_files:
        image_path = os.path.join(input_image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Preskocenie {image_file}: Obrazok alebo maska nebola najdena.")
            continue

        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        output = np.zeros_like(image, dtype=np.uint8)
        output[binary_mask == 255] = image[binary_mask == 255]

        angle = random.uniform(-30, 30)
        rotated_image, rotated_mask = rotate_image_and_mask(output, binary_mask, angle)

        width_scale = random.uniform(2.0, 3.0)
        height_scale = random.uniform(0.5, 1.0)

        new_width = int(rotated_image.shape[1] * width_scale)
        new_height = int(rotated_image.shape[0] * height_scale)

        resized_image = cv2.resize(rotated_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized_mask =  cv2.resize(rotated_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


        output_path = os.path.join(output_dir, f"extracted_{image_file}")
        mask_output_path = os.path.join(output_dir, f"mask_{image_file}")

        cv2.imwrite(output_path, resized_image)
        cv2.imwrite(mask_output_path, resized_mask)

        print(f"Ulozeny extrahovany objekt: {output_path}")
        print(f"Ulozena nova maska: {mask_output_path}")

    print("Hotova extrakcia.")



if __name__=="__main__":
    main()