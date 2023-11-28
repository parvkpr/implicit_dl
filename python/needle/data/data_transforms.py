import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            flipped_img = np.fliplr(img)
        else:
            flipped_img = img

        return flipped_img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)


        # Crop the padded image based on the shift values
        crop_x1 = self.padding+shift_x
        crop_x2 = crop_x1+ img.shape[0]
        crop_y1 = self.padding+shift_y
        crop_y2 = crop_y1+ img.shape[1]

        # Crop the padded image based on the calculated bounds
        cropped_img = padded_img[crop_x1:crop_x2, crop_y1:crop_y2, :]

        # Ensure the cropped image has the same dimensions as the original image
        #cropped_img = cropped_img[:h, :w, :]

        return cropped_img

        ### END YOUR SOLUTION
