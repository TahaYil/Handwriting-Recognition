from sympy import false
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image,ImageOps,ImageEnhance
import torch

# Model ve işlemciyi yükle
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten',use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

#TÜM GÖRSELDE İŞE YARAMIYOR SATIR SATIR
image = Image.open("tahacontrol.png")

# GRİYE ÇEVİRME
gray_image = image.convert("L")

# TRESHOLD
threshold_image = gray_image.point(lambda x: 0 if x < 165 else 255, '1')

# RGB ÇEVİRME
threshold_rgb = threshold_image.convert("RGB")

threshold_rgb.show()
#image=image.convert("RGB")

# Modelle işlem yaparken kullanmak için:
pixel_values = processor(images=threshold_rgb, return_tensors="pt").pixel_values

# Tahmin
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🖋️ Tanınan Metin:\n")
print(generated_text)
print("elif harika bir insan")




