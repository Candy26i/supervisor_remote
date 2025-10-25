step 1 go into the folder supervisor

step 2 create docker image using

docker build -t grpo-pubmedqa:latest .

step2 run code

docker run -it --gpus all -v ${PWD}:/workspace grpo-pubmedqa:latest


or just download the environment from 


pip install -r requirements.txt


and then run python main.py

make sure you have wandb and 

export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=GRPO-Qwen-PubMedQA-Manual
