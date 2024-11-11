import os
from datasets import Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import evaluate
import torch

# 1. 로컬 이미지 경로 설정
image_dir = "./images"  # 로컬 이미지 경로
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]  # 이미지 파일 리스트

# 2. 라벨 정의 (이미지와 라벨 매칭)
labels = ["cat", "person", "dog"]  # 라벨을 필요에 맞게 수정

# 3. 이미지 로드
images = [Image.open(image_path) for image_path in image_paths]

# 4. 로컬 데이터를 Hugging Face Dataset으로 변환
data = {"image": images, "label": labels * (len(images) // len(labels))}  # 라벨을 이미지 수에 맞게 반복
dataset = Dataset.from_dict(data)

# 5. 모델과 프로세서 로드
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# 6. 이미지를 모델에 맞게 처리
inputs = processor(images=images, text=labels, return_tensors="pt")
print("input_ids:", inputs["input_ids"])
print("attention_mask:", inputs["attention_mask"])
print("pixel_values:", inputs["pixel_values"])
print("image_shape:", inputs["pixel_values"].shape)

# 7. 모델 평가
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("outputs:", outputs.keys())
    print("logits_per_image:", logits_per_image)
    print("probs:", probs)

for idx, prob in enumerate(probs):
    print(f"- Image #{idx}")
    for label, p in zip(labels, prob):
        print(f"{label}: {p:.4f}")

# 8. DataLoader로 테스트 데이터셋 준비
test_dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=lambda batch: (
        [item["image"] for item in batch],
        [item["label"] for item in batch],
    ),
)

# 9. 정확도 평가
metric = evaluate.load("accuracy")
predictions, references = [], []
labels_names = labels  # 로컬 라벨 사용

# 라벨을 정수 인덱스로 변환
label_map = {label: idx for idx, label in enumerate(labels_names)}

model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        inputs = processor(images=images, text=labels_names, return_tensors="pt")
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        # predictions는 인덱스 값이므로 argmax를 사용
        predictions += probs.argmax(dim=1).cpu().tolist()

        # references는 라벨을 인덱스로 변환
        references += [label_map[label] for label in labels]

results = metric.compute(predictions=predictions, references=references)
print(f"클래스 목록: {labels_names}")
print(f"정확도: {results['accuracy'] * 100:.2f}%")




'''
로컬 이미지 파일을 사용할 때 고려할 점:
이미지 형식: CLIPProcessor는 PIL 이미지 또는 URL 형식의 이미지를 입력으로 받습니다. 따라서 로컬 이미지 파일을 사용하려면 이를 PIL 이미지 객체로 열어야 합니다. PIL.Image.open()을 사용하면 로컬 이미지 파일을 PIL 이미지 객체로 변환할 수 있습니다.

라벨: 로컬 이미지의 경우, 이미지 파일 이름이나 별도의 텍스트 파일을 통해 라벨을 매칭해야 할 수 있습니다. 라벨을 어떻게 정의할지는 데이터셋에 맞게 설정해야 합니다.

파일 경로: 로컬 파일 경로를 정확하게 지정해야 합니다. 파일 경로가 잘못되면 이미지가 로드되지 않거나 오류가 발생할 수 있습니다.

따라서, 로컬에 있는 이미지 파일을 사용하려면 이미지를 직접 PIL.Image로 읽어들여 CLIPProcessor에 입력해야 하며, 이를 위해 datasets 라이브러리 대신 일반적인 파일 I/O 작업을 사용할 수 있습니다.
'''