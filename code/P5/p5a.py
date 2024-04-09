from PIL import Image
import numpy as np

class ReducingresolutionClass:
    def __init__(self, n=2):
        self.n = n

    def reduce(self, images):
        low_res_images = [img.resize((img.width // self.n, img.height // self.n)) for img in images]
        # Create the dataset
        features = []
        labels = []

        end_of_each_image = {}
        k = 1

        for img, low_res_img in zip(images, low_res_images):
            for i in range(0, img.width):
                for j in range(0, img.height):
                    # Get the corresponding pixel and its eight neighbors in the lower-resolution image
                    i_in_low_res_img = i // self.n
                    j_in_low_res_img = j // self.n
                    neighbors = self.get_neighbors(low_res_img, i_in_low_res_img, j_in_low_res_img)

                    # Create a feature vector of 27 elements
                    feature = np.array(neighbors).flatten()
                    features.append(feature)

                    # The label is the 3 color channels of the corresponding pixel in the original image
                    label = np.array(img.getpixel((i, j)))
                    labels.append(label)
            end_of_each_image[k] = len(features)
            k += 1


        return features, labels, end_of_each_image, low_res_images

    def get_neighbors(self, img, i, j):
        neighbors = []
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                try:
                    pixel = img.getpixel((x, y))
                    neighbors.extend(pixel)  # Flatten the pixel values before appending
                except:
                    neighbors.extend([0, 0, 0])  # Append a black pixel if out of bounds
        return neighbors
