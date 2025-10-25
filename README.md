Using Docker

Step 1: Go into the project folder:

cd supervisor


Step 2: Build the Docker image:

docker build -t grpo-pubmedqa:latest .


Step 3: Run the container with GPU support:

docker run -it --gpus all -v ${PWD}:/workspace grpo-pubmedqa:latest

Option 2 — Run Locally (Without Docker)

Step 1: Install dependencies:

pip install -r requirements.txt


Step 2: Set up Weights & Biases (W&B) for experiment tracking:

export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=GRPO-Qwen-PubMedQA-Manual


Step 3: Run the training script:

python main.py
