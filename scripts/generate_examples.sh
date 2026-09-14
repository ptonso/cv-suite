#!/bin/bash
set -e

DATA="docs/example_data"

echo "Cleaning up old data..."
rm -rf $DATA/

echo "Setting up example data..."
mkdir -p $DATA/images
mkdir -p $DATA/dump

# Download sample images (using reliable direct image URLs)
echo "Downloading sample images..."
# Fire (public-domain campfire photo, so grounding on "fire" actually has fire to find)
curl -sL "https://upload.wikimedia.org/wikipedia/commons/8/82/Campfire_flames.jpg" -o $DATA/images/fire.jpg
# Dog (using the reliable PyTorch hub dog image)
curl -sL "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" -o $DATA/images/dog.jpg

# Create a mock YOLO dataset for the conversion example
mkdir -p $DATA/dataset-yolo/images/train $DATA/dataset-yolo/labels/train
cp $DATA/images/dog.jpg $DATA/dataset-yolo/images/train/
echo "0 0.5 0.5 1.0 1.0" > $DATA/dataset-yolo/labels/train/dog.txt
cat << 'EOF' > $DATA/dataset-yolo/dataset.yaml
path: .
train: images/train
val: images/train
names:
  0: dog
EOF

# Create a mock dump for the prep example
cp $DATA/images/dog.jpg $DATA/dump/img1.JPG
cp $DATA/images/dog.jpg $DATA/dump/img1_duplicate.JPG # duplicate to show deduplication
mkdir -p $DATA/dump/subfolder
cp $DATA/images/fire.jpg $DATA/dump/subfolder/img2.JPEG

echo "----------------------------------------"
echo "Running cvsuite examples from README..."
echo "----------------------------------------"

# 1. Convert formats
echo "1. Format conversion (YOLO -> COCO)"
cvsuite label ./$DATA/dataset-yolo to-coco ./$DATA/dataset-coco

# 2. Auto-label (Grounding)
echo -e "\n2. Zero-shot grounding"
cvsuite label ./$DATA/images ground --device cuda --provider gsam --prompt "fire" to-results ./$DATA/labeled

# 3. VLM Captioning
echo -e "\n3. VLM captioning"
cvsuite vlm ./$DATA/images caption --provider qwen --precision bf16 --model-arg max_pixels=2007040 to-json ./$DATA/captions

# 4. Zero-shot classification
echo -e "\n4. Zero-shot classification"
cvsuite class ./$DATA/images infer --provider clip --prompt "good,bad" to-class-dir ./$DATA/sorted

# 5. Image Generation
echo -e "\n5. Image Generation"
cvsuite gen create --provider stable_diffusion --precision fp16 --prompt "a red fox in the snow" to-dst ./$DATA/fox.png

# 6. Prep
echo -e "\n6. Folder preparation (flatten, dedup, rename)"
cvsuite prep arrange ./$DATA/dump ./$DATA/clean --unzip --exact-dedup --rename-seq

echo -e "\nDone! All examples generated in ./$DATA/"
