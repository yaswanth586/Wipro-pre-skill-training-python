# Applying Transformations using torchvision.transforms
# Problem 1: Load an image using PIL, convert it to a PyTorch tensor, and normalize it using torchvision.transforms.
# Normalize with mean=0.5 and std=0.5.
# Problem 2: Create a custom transformation that rotates an image by 45 degrees and apply it to an image.
# Use torchvision.transforms.Compose to chain this custom transformation with a resize transformation that resizes
# the image to 128x128 pixels.
# Problem 3: Use torchvision.transforms to apply the following transformations to an image:
# Random horizontal flip.
# Random crop of size 100x100.
# Convert the image to grayscale.


from PIL import Image
import torchvision.transforms as transforms

# Load an image using PIL
image_path = r'./img.jpg'
image = Image.open(image_path)

# Problem 1: Load an image, convert it to a PyTorch tensor, and normalize it
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming the image is grayscale
])

# Apply the transformation
tensor_image = transform1(image)
print("Problem 1: Normalized Tensor Image")
print(tensor_image)


# Problem 2: Create a custom transformation to rotate by 45 degrees and resize to 128x128
class RotateTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return x.rotate(self.angle)


transform2 = transforms.Compose([
    RotateTransform(45),
    transforms.Resize((128, 128))
])

# Apply the transformation
transformed_image2 = transform2(image)
transformed_image2.show(title="Problem 2: Rotated and Resized Image")

# Problem 3: Apply random horizontal flip, random crop, and convert to grayscale
transform3 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((100, 100)),
    transforms.Grayscale()
])

# Apply the transformations
transformed_image3 = transform3(image)
transformed_image3.show(title="Problem 3: Random Flip, Crop, and Grayscale")
