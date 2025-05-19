from codes.other_functions.my_functions import os, cv2, shuffle_files, resize_file, blend_images

def main():
    #Paths for input and output images and masks
    old_images_path = '../DP2/generated_images_and_masks_2.1/extr_images'
    masks_path = '../DP2/generated_images_and_masks_2.1/extr_masks'
    new_images_path = '../DP2/generated_images_and_masks_2.1/first_images'
    output_path = '../DP2/generated_images_and_masks_2.1/final_images'
    masks_output_path = '../DP2/generated_images_and_masks_2.1/final_images_masks'

    mask_filenames = []
    new_images = []

    shuffle_files(mask_filenames, masks_path)
    shuffle_files(new_images, new_images_path)

    # Loop over the images
    for new_image_index in range(len(new_images)):
        new_image_filename = new_images[new_image_index]

        # Stop if there are not enough masks for all new images
        if new_image_index >= len(mask_filenames):
            print("Nedostatok starych fotiek/masiek na pokrytie vsetkych novych obrazkov.")
            break

        mask_filename = mask_filenames[new_image_index]

        # Create full paths to old image, mask, and new image
        old_image_path = os.path.join(old_images_path, mask_filename)
        mask_image_path = os.path.join(masks_path, mask_filename)
        new_image_path = os.path.join(new_images_path, new_image_filename)

        # Load the images
        old_image = cv2.imread(old_image_path, cv2.IMREAD_UNCHANGED)
        old_mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        new_image = cv2.imread(new_image_path, cv2.IMREAD_UNCHANGED)

        # Skip processing if any image failed to load
        if old_image is None or old_mask is None or new_image is None:
            print(f"Preskocenie, error, nepodarilo sa nacitat: {mask_filename} or {new_image_filename}")
            continue

        # Resize old image and mask to match new image dimensions
        old_image_resized = resize_file(old_image, new_image.shape)
        old_mask_resized = resize_file(old_mask, new_image.shape)

        # Blend old and new images using the resized mask
        result_image = blend_images(old_image_resized, new_image, old_mask_resized)

        # Define output paths for the blended image and its mask
        output_image_path = os.path.join(output_path, new_image_filename)
        mask_output_path = os.path.join(masks_output_path, new_image_filename)

        # Save the image and new mask
        cv2.imwrite(output_image_path, result_image)
        cv2.imwrite(mask_output_path, cv2.resize(old_mask, (new_image.shape[1], new_image.shape[0])))

        # Saved files
        print(f"Ulozeny: {output_image_path}")
        print(f"Ulozena maska do: {mask_output_path}")



if __name__=="__main__":
    main()