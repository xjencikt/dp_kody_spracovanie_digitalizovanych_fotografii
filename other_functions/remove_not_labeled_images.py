from codes.other_functions.my_functions import *

def main():
    # Define image, mask folder and function to remove images, that do not have masks made for them
    images_folder = 'D:/v3.0/more_images'
    masks_folder = 'D:/v3.0/more_masks'
    remove_images_without_masks(images_folder, masks_folder)

if __name__ == '__main__':
    main()