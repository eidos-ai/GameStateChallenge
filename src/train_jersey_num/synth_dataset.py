
# import os
# import argparse
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from torchvision import transforms
import os


class BlurImage:
    def __init__(self, radius=5):
        self.radius = radius
    
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))
        
class AddRandomNumberTransform:
    def __init__(self, radius=5):
        self.radius = radius
        self.fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
        ]
        self.random_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.3, 1.3)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

    def __call__(self, img):
        # Blur the input image
        pregenerate_transforms = transforms.Compose([            
            # transforms.RandomAffine(degrees=45, translate=(0, 1), scale=(2, 3)),
            BlurImage(radius=7)
        ])
        background = pregenerate_transforms(img)
        
        # Create a transparent image to draw the number
        number_image = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(number_image)

        # Select a random font
        font_path = random.choice(self.fonts)
        font = ImageFont.truetype(font_path, 80)

        # Generate a random number
        number = random.randint(1, 99)
        text = str(number)

        # Calculate text position for centering
        width, height = img.size
        bbox = d.textbbox((width // 2, height // 2), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2

        # Random color for the number
        fill_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 255)

        # Draw the text
        d.text((x, y), text, font=font, fill=fill_color)

        # Apply transformations to the number image
        transformed_number_image = self.random_transforms(number_image)

        # Blend the number image onto the blurred image
        blended_image = Image.alpha_composite(background.convert('RGBA'), transformed_number_image)
        blended_image = blended_image.convert('RGB')

    
        return blended_image, number
        
class SnythNumberDatasetGenerator(Dataset):

    def __init__(self, root_dataset: Dataset):
        self.root_dataset = root_dataset
        self.generate = AddRandomNumberTransform(radius=6)
        
    def __len__(self):
        return len(self.root_dataset)
        
    def __getitem__(self, idx):
        original_img, original_label = self.root_dataset[idx]
        synth_img, snyth_number = self.generate(original_img)
        
        # Convert PIL image to Tensor
        # totensor = transforms.ToTensor()
        # synth_img_tensor = totensor(synth_img)
        # resize = transforms.CenterCrop(size=(224, 224))
        # synth_img_tensor = resize(synth_img_tensor)
        return synth_img, snyth_number


class SnythNumberDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.classes = [d.name for d in os.scandir(main_dir) if d.is_dir()]
        self.classes.sort()
        self.images = []
        for cls in self.classes:
            cls_folder = os.path.join(main_dir, cls)
            for img_filename in os.listdir(cls_folder):
                if img_filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    self.images.append((os.path.join(cls_folder, img_filename), int(cls)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)
