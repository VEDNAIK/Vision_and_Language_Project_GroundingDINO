# Vision_and_Language_Project_GroundingDINO

# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

**GroundingDINO** is a state-of-the-art text-guided object detection and phrase grounding model that uses advanced vision-language alignment mechanisms for accurate and efficient object detection.

## Features

- **Text-Guided Object Detection:** Detect objects in an image based on textual input queries.
- **Pre-trained Models:** High-performance pre-trained models available for use.
- **Custom Training:** Easily train the model on new datasets.
- **Evaluation Pipeline:** Comprehensive evaluation metrics and subset-focused testing capabilities.

---


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   ```

2.	Set Up Python Environment:

  	•	Create a virtual environment:
    ```bash
     git clone https://github.com/IDEA-Research/GroundingDINO.git
     cd GroundingDINO
    ```
    
    •	Install dependencies:    
    ```bash
        pip install -r requirements.txt
    ```

## 2. Set Up Python Environment

To ensure a proper environment for running GroundingDINO, follow these steps:

1. **Create a Virtual Environment:**
   Use Python's built-in `venv` module to create an isolated environment for the project:
   ```bash
   python3 -m venv groundingdino_env
   source groundingdino_env/bin/activate
   ```

2.	Install PyTorch:
    Install the compatible PyTorch version for your system:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    ```

3.	Set Up COCO Dataset:
Download the COCO dataset from COCO’s website and organize it as follows:

    ```
    dataset/
        coco/
            train2017/
            val2017/
            annotations/
    ```
    

4.	Download Pre-trained Weights:
	•	Download the pre-trained GroundingDINO weights:

    ```bash
    wget https://path/to/pretrained_weights.pth -O weights/groundingdino_pretrained.pth
    ```

   
## 3. Usage

1. Testing Pre-trained Model

    Run the pre-trained GroundingDINO model on the COCO dataset to evaluate its performance:
    
      ```bash
        python tools/test.py \
            --config configs/groundingdino_coco.yaml \
            --weights weights/groundingdino_pretrained.pth \
            --dataset dataset/coco/val2017 \
            --output results/
      ```
    Expected Output:
    
    •	mAP and Recall metrics for text-guided detection tasks will be displayed.
   
    •	Results are saved in the results/ folder, including visualizations.

3. Training from Scratch
  
    To train GroundingDINO on the COCO dataset from scratch, use the following command:
    
     ```bash
    python tools/train.py \
        --config configs/groundingdino_coco.yaml \
        --dataset dataset/coco/train2017 \
        --output_dir checkpoints/
     ```
     
    Key Training Parameters:
    •	Batch Size: Default is 16. Modify in the configs/groundingdino_coco.yaml file.
    •	Learning Rate Warm-Up: Automatically applied as part of the optimizer setup.

3. Custom Dataset Training

    To train on a custom dataset:
   
   1.	Convert your dataset to COCO format.
   2.	Update the configs/custom_dataset.yaml file with the appropriate paths and classes.
   3.	Run the training script:
    
     ```bash
    python tools/train.py \
        --config configs/custom_dataset.yaml \
        --output_dir checkpoints_custom/
     ```

4. Running Subset-Focused Evaluation

    Evaluate the model on specific subsets (e.g., rare objects or complex phrases):
    
     ```bash
    python tools/eval.py \
        --config configs/groundingdino_coco.yaml \
        --weights weights/groundingdino_pretrained.pth \
        --subset rare_objects
     ```

 
## 4. Advanced Features

  Data Augmentation
  
  Advanced augmentations like MixUp and CutOut can be enabled by modifying configs/groundingdino_coco.yaml:
  
   ```yaml
  augmentation:
    mixup: true
    cutout: true
    color_jitter: true
   ```

  Custom Loss Functions
  
  Weighted focal loss and IoU-aware loss can be enabled by updating the loss_function section in configs/groundingdino_coco.yaml:
  
   ```yaml
  loss_function:
    classification: weighted_focal_loss
    regression: iou_loss
   ```

## File Structure

   ```plaintext
  GroundingDINO/
  ├── configs/                # Configuration files for datasets and models
  ├── datasets/               # Dataset loaders and utilities
  ├── models/                 # Model architecture and loss functions
  ├── tools/                  # Scripts for training, testing, and evaluation
  ├── weights/                # Pre-trained weights
  ├── requirements.txt        # Dependency list
  └── README.md               # Documentation
   ```





