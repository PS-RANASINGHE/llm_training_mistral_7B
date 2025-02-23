# Explaination of the code 
For this project I used the google's infra 'google Colab' to host and run the project. Let us go through.

Import the following libraries 

```py
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets matplotlib
```
Next use the Huggingface login (I used the CLI login but the notebook option is also possible).

```py
from huggingface_hub import notebook_login
!huggingface-cli login
```
Apply the token of the Huggingface 


```py
  _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: fineGrained).
The token `llmtraining` has been saved to /root/.cache/huggingface/stored_tokens
Your token has been saved to /root/.cache/huggingface/token
Login successful.
The current active token is: `llmtraining`
```

Load the Dataset 

```py
from datasets import load_dataset

train_dataset = load_dataset('json', data_files='/content/llm_training_mistral_7B/go.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='/content/llm_training_mistral_7B/cleaned_tosibox_validation_datas.jsonl', split='train')
```

Here I am using `Accelerate`  and Fully Sharded Data Parallel (FSDP) from `PyTorch` to efficiently train large language model.

Using `fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False), optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))` I created the the `FSDP` plugin for Accelerate. This line initializes the FSDP plugin that will be passed to the Accelerator. It controls how the model's parameters and optimizer state are distributed across multiple devices.

-  `state_dict_config`: Configures how the model's weights (state dict) are handled.
-  `offload_to_cpu=True`: This tells the system to offload the model weights to the CPU whenever they are not needed on the GPU, thus reducing GPU memory usage. This is particularly useful when training large models like Mistral 7B and Mistral 24B, which consume large amounts of memory.
-  `rank0_only=False` : By default, only the rank0 worker (the main worker in a distributed setting) handles the model's state dictionary. Setting rank0_only=False means the model's state dict is sharded across all workers to distribute the memory load.
-  `optim_state_dict_config`: This configures the optimizer’s state dictionary similarly to how the model's weights are handled.
-  



`Accelerate` is a Hugging Face library that simplifies distributed training, especially for complex setups like multi-GPU and multi-node training. It abstracts much of the boilerplate code, allowing you to focus on model training. `FSDP` is a technique used in distributed training that helps in efficiently sharding the model across multiple devices. This can significantly reduce the memory usage and improve training efficiency, especially when dealing with large models such as Mistral 7B and 24B.



```py
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```


The following code is for monitoring the performace of the model.

`Weights & Biases (W&B or wandb)` is a tool designed to help track, visualize, and manage machine learning experiments. It provides a platform to log metrics, hyperparameters, and other details during model training. With W&B, you can easily monitor training progress in real-time, compare different runs, and visualize results through interactive dashboards. It also allows for collaboration, model versioning, and experiment tracking, making it easier to manage complex machine learning workflows and share results with teams or stakeholders.

I start by installing the `wandb` library using pip with the `-q` flag for quiet installation, ensuring the installation is done without unnecessary output. I then `import wandb` and os to interact with the Weights & Biases (W&B) platform and set environment variables. Next, I use `wandb.login()` to authenticate and establish a connection to my W&B account. I define the project name `journal-finetune` as the value for the `wandb_project` variable, and if the project name is `non-empty`, I set the environment variable `WANDB_PROJECT` to this project name. This allows W&B to track and log the training process under the specified project, enabling me to monitor metrics, visualizations, and manage experiments in the W&B dashboard.


```py
!pip install -q wandb -U

import wandb, os
wandb.login()

wandb_project = "journal-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
```



```py
def formatting_func(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return text
```


Here the above function `formatting_func` takes an example, which is a dictionary containing a input and output, and formats it into a string that displays the question and answer in a structured format. Specifically, it outputs the text in a style where the question is prefixed with `"### Question"` and the answer with `"### Answer"`. This helps present the data in a readable and organized manner. Additionally, the code checks if there is a specified `wandb_project`, and if so sets the environment variable `WANDB_PROJECT` to the value of `wandb_project`. This allows me to specify the project for Weights & Biases logging, ensuring that the experiment is tracked under the correct project in the wandb platform.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

```


Next I am using PyTorch and the Transformers library to load a LLM specifically Mistral-7B-v0.1 from the Hugging Face model hub. First I define a configuration for `4-bit` quantization using BitsAndBytesConfig to optimize memory usage and computation. The configuration includes settings like loading the model in `4-bit precision` (load_in_4bit=True), using double quantization `bnb_4bit_use_double_quant=True` and specifying the quantization type as `NF4`. Additionally I set the compute data type for the quantized weights to `bfloat16` to strike a balance between performance and precision. Then I load the model using `AutoModelForCausalLM.from_pretrained()` passing the quantization configuration `bnb_config` and using `device_map="auto"` to automatically place the model across available devices, ensuring efficient resource utilization while maintaining low memory footprint.



| **Precision**    | **Memory Usage**       | **Training Speed**      | **Precision/Accuracy**    | **Typical Use Case**                                | **Hardware Support**       | **Use in Large Models**      |
|------------------|------------------------|-------------------------|---------------------------|-----------------------------------------------------|----------------------------|-----------------------------|
| **FP32**         | High (32 bits)         | Slower                  | High                      | Training large models, ensuring stability          | All GPUs, TPUs             | Essential for stability     |
| **FP16**         | Medium (16 bits)       | Faster                  | Moderate                  | Mixed Precision, GPU/TPU accelerated tasks          | NVIDIA GPUs, A100, V100    | Suitable for large models   |
| **Bfloat16**     | Medium (16 bits)       | Fastest (on TPUs)       | Moderate                  | Efficient large model training on TPUs             | TPUs (Google Cloud)        | Optimized for large models  |
| **BF4**          | Very Low (4 bits)      | Very Fast               | Low                       | Extreme memory-efficient, quantized models         | Limited support (NVIDIA)   | Used for extremely large models|
| **INT8**         | Very Low (8 bits)      | Fast                    | Low                       | Quantized inference on edge devices, Mobile, etc.  | NVIDIA Tensor Cores, Edge Devices | Suitable for smaller models |
| **FP64**         | Very High (64 bits)    | Slow                    | Very High                 | Scientific computing, High precision tasks         | High-performance servers   | Rarely used in LLMs        |
| **Mixed Precision (FP16 & FP32)** | Low (due to FP16) | Faster (due to FP16) | High (using FP32 for some operations) | Accelerating training while maintaining accuracy  | All GPUs (especially NVIDIA Tensor Cores) | Ideal for large models requiring stability |
| **INT4**         | Extremely Low (4 bits) | Extremely Fast          | Very Low                  | Ultra-low precision for inference                  | Limited support            | For highly compressed models |
