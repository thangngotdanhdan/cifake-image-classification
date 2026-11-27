
from PIL import Image, ImageOps
from torchvision import transforms

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image):
    """
    Preprocesses an image for the trained ResNet18 model.

    Accepts either:
    - a PIL image
    - a filepath string

    Returns:
    - tensor with shape [1, 3, 224, 224]
    """

    if isinstance(image, str):
        image = Image.open(image)

    image = ImageOps.exif_transpose(image).convert("RGB")

    image_tensor = RESNET_TRANSFORM(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
