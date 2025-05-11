from sympy import false
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageEnhance
import torch
import cv2
import numpy as np

# Model ve işlemciyi yükle
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Görüntüyü yükle
image = Image.open("denedene.png")

# Görüntüyü numpy dizisine dönüştür
image_array = np.array(image)
grayImage = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

# Otsu Thresholding uygula
# Otsu, eşik değerini otomatik olarak belirler
ret, otsu_thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morfolojik işlemler - harflerdeki kopmaları birleştirmek için
# 1. Kapama (Closing) işlemi - kopuk harfleri birleştirir
kernel = np.ones((2, 2), np.uint8)  # Kernel boyutunu metninize göre ayarlayabilirsiniz
closing = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)

# 2. Genişletme (Dilation) işlemi - harfleri daha kalın yapar, boşlukları doldurur
dilation_kernel = np.ones((1, 1), np.uint8)
dilated_image = cv2.dilate(closing, dilation_kernel, iterations=1)

# 3. Median filtreleme - gürültüyü azaltır
filtered_image = cv2.medianBlur(dilated_image, 3)

# Numpy dizisini PIL Image'e dönüştür
threshold_image = Image.fromarray(otsu_thresh)
threshold_rgb = threshold_image.convert('RGB')

# Görüntüyü göster
threshold_rgb.show()

# Modelle işlem için görüntüyü hazırla
pixel_values = processor(images=threshold_rgb, return_tensors="pt").pixel_values

# Tahmin yap
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🖋️ Tanınan Metin:\n")
print(generated_text)