
#### Note - Saved model can be reveiwed from https://huggingface.co/sesanaa/mistral-journal-finetune 

Note - As for some reasons I have make this model private. In order to download the model there should be a token generated from the user. 

In this project I have used Mistral AI 7B as the base model and a dataset that is used to finetune.

Why Mistral AI is Better for Fine-Tuning:
1. Open-Weight Models – Mistral AI releases its models with open weights, allowing full customization and fine-tuning without restrictions. This makes it an ideal choice for developers who need to adapt models to specific tasks or industries.
2. High Performance with Fewer Parameters – Despite having fewer parameters, Mistral models achieve performance comparable to or superior to larger models like LLaMA. This efficiency translates to faster training, lower costs, and reduced hardware requirements.
3. Advanced Attention Mechanisms – Mistral integrates optimizations such as Grouped Query Attention (GQA) and Sliding Window Attention (SWA), improving both inference speed and the model’s ability to process longer sequences, which is crucial for fine-tuning tasks.
4. Portability and Deployment Versatility – Mistral models can be deployed across various environments, including cloud services, local servers, and on-premise hardware. This flexibility makes them an attractive choice for businesses with strict data governance needs.
5. Optimized for Resource Efficiency – Fine-tuning Mistral models requires fewer computational resources compared to LLaMA models with similar capabilities. This reduces operational costs while maintaining strong language generation performance.

In here i have manually tested how the model performs in generation of answers for user questions before and after fine tuning. Also another method that I can use to improve performance is Retrieval-Augmented Generation (RAG).

**Retrieval-Augmented Generation (RAG)**

RAG enhances language models by integrating an external retrieval mechanism, allowing dynamic access to relevant knowledge.

### **Benefits of RAG**
- **Better Domain-Specific Accuracy**: Model can fetch the latest data rather than relying on outdated parameters.
- **Reduced Hallucinations**: Answers are grounded in retrieved documents, improving reliability.
- **Scalability**: Works across multiple domains without requiring new fine-tuning.
- **Lower Compute Cost**: No need for large-scale retraining; retrieval is handled separately.

---

### Fine-Tuning vs. RAG: A Comparison**

| Feature           | Fine-Tuning | RAG |
|------------------|------------|-----|
| **Customization** | Static model trained on a dataset | Dynamic retrieval of external knowledge |
| **Resource Cost** | High (GPU-intensive) | Lower (retrieves at runtime) |
| **Flexibility** | Requires retraining for updates | Can update knowledge dynamically |
| **Performance** | Strong for specific tasks | Versatile, can adapt to multiple domains |
| **Latency** | Low once trained | Slightly higher due to retrieval process |

**💡 Best Practice:** Use a **hybrid approach** by fine-tuning for core improvements and using RAG for real-time knowledge retrieval.

Basically in RAG what it does is retrieves the exact data from somewhere(probably a database or wifi page) does some content scraping and generate nice texts as it has trained before. This is better where you need an accurate 100% output everytime. 

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




```py
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))
```

Here I am using the `AutoTokenizer` from Hugging Face's Transformers library to load a pre-trained tokenizer for my model using the `base_model_id`. I set the tokenizer to pad sequences on the left `padding_side="left"` to match the model's expected input format. Additionally I add the End-of-Sequence (EOS) token and the Beginning-of-Sequence (BOS) token to the tokenizer to handle the start and end of each input sequence appropriately. After loading the tokenizer I assign the EOS token as the padding token `tokenizer.pad_token = tokenizer.eos_token` which ensures that any padding tokens used during batching are consistent with the EOS token. Lastly I define a function `generate_and_tokenize_prompt` which takes a prompt, applies a formatting function to it and then tokenizes the result using the previously defined tokenizer. This function ensures that any input prompt is properly formatted and tokenized before being fed into the model for generation.


```py
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
```
I use the `map()` function to process both the training and validation datasets. I apply `generate_and_tokenize_prompt` to each dataset. This function is responsible for transforming the raw input data into tokenized prompts that are suitable for training the model. By using `tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)` I make sure that each example in the training dataset is tokenized and prepared for model input.

In summary here i have set the `max_length` to 512, as it was an appropriate length for my dataset. The function `generate_and_tokenize_prompt2` takes a prompt applies a formatting function to it and then passes it through the tokenizer. Inside the tokenizer I enabled truncation to ensure that input sequences longer than 512 tokens are shortened while any sequences shorter than 512 tokens are padded to the maximum length. Additionally I assign the `input_ids` to the labels field. This is typically done for tasks like language modeling where the model learns to predict the next token in the sequence. Finally the function returns the tokenized result, ready for input into the model for training or inference. Then the 2 datasets are the mapped as previous accirding to this function 


```py
max_length = 512 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)
```
```py
eval_prompt = " I want to use device 175. What are the specification of it # "
# Init an eval tokenizer that doesn't add padding or eos token
eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
)

model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True))
```

Before I start training i want to check how the base model works. So I sent a basic prompt as mentioned above. In this code I initialize an evaluation tokenizer using the `AutoTokenizer` class from Hugging Face's transformers library making sure it doesn't add any padding or EOS tokens by setting `add_bos_token=True`. This is crucial because I don't want additional tokens during the evaluation phase, ensuring the input sequence is processed as intended. After that, I tokenize the evaluation prompt, convert it into tensor format and move it to the GPU using `.to("cuda")` to speed up the computation. I set the model into evaluation mode using `model.eval()` which disables certain training-specific operations like dropout. Then I generate the model’s output by passing the tokenized input to the model with model.generate(). I also specify a max_new_tokens=256 limit for the generated text and a `repetition_penalty=1.15` to discourage repetitive text generation. After the model generates the output I decode the generated tokens back into text using the tokenizer's `decode()` function, and `skip_special_tokens=True` ensures that any special tokens like padding or EOS tokens are excluded from the final output. Finally, the generated output is printed, providing the result of the evaluation prompt in a human-readable form.

So the output generated is as follows. It is clear that it is not even close to been usable.

```cmd
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
I want to use tosibox 175. What are the specification of it # 2018-04-19

I am using a Toshiba Satellite L635D-S7000 laptop with Windows 7 Home Premium (64 bit). The computer is about two years old and has been working fine until recently when I started getting an error message that says "Windows cannot access the specified device, path or file. You may not have the appropriate permissions to access the item." This happens whenever I try to open any program on my computer. I have tried restarting my computer several times but this does not fix the problem. I also tried running a system scan for viruses but nothing was found. Any help would be greatly appreciated!

```
```py
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```
I am enabling gradient checkpointing and preparing the model for `k-bit` training using the **PEFT (Parameter Efficient Fine-Tuning)** library. First I call `model.gradient_checkpointing_enable()` to enable gradient checkpointing, a technique that helps save memory during training. By not storing intermediate activations during the forward pass and instead recomputing them during the backward pass, I can significantly reduce GPU memory usage which is crucial when training large models. Then I use the function `prepare_model_for_kbit_training(model)` from **PEFT**. This prepares the model for **k-bit training** which reduces the memory footprint by storing model weights in lower precision formats like `8-bit` or `4-bit`. PEFT is a library designed to enable fine-tuning large models more efficiently, without the need to fine-tune all the parameters. It offers methods for tuning only a small subset of parameters while keeping the rest frozen, which reduces computational cost and memory usage. By using techniques like low-bit quantization `k-bit precision` **PEFT** allows for faster training lower memory consumption and more efficient use of resources when working with large models. This is especially important for deploying large language models like Mistral 7B or similar models in resource-constrained environments where optimizing memory and computational efficiency is key.

```py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```
I define `print_trainable_parameters(model)` which is designed to calculate and display the number of trainable parameters in a model as well as the total number of parameters. The function starts by initializing two variables: trainable_params to track the number of parameters that require gradients and all_param to track the total number of parameters in the model. Then I iterate over each parameter in the model using `model.named_parameters()` which returns both the name and the parameter tensor. For each parameter, I add its size to all_param, and if the parameter requires gradients (ie it is trainable) I also add its size to `trainable_params`. Finally I print the results: the number of trainable parameters, the total number of parameters, and the percentage of parameters that are trainable, calculated as the ratio of `trainable_params` to `all_param` multiplied by 100. This gives a useful overview of how much of the model is trainable during the fine-tuning process.

This is useful for me to get an idea what I can train and what are the trainable parameters in our model.

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
```
In this code, I’m using **LoRA (Low-Rank Adaptation)**, a technique designed to make fine-tuning large models more efficient by introducing low-rank matrices into certain layers of the model. LoRA reduces the memory and computational burden during fine-tuning by allowing us to update only a small set of parameters, rather than the entire model. Specifically, **LoRA** adds low-rank adapters to pre-trained models, focusing on the weights of specific modules. In this case, I specify the `target_modules`, such as `q_proj`, `k_proj`, `v_proj`, and others, which are key components of the model’s attention mechanism. By applying LoRA to these layers, only the low-rank matrices in these modules will be trained, while the rest of the model's parameters remain frozen, leading to faster and more memory-efficient training. 

The **LoRA configuration** is set with `r=32`, determining the rank of the low-rank matrices, which controls the amount of learned information during fine-tuning. A higher rank allows the model to adapt more complex patterns, while a lower rank keeps the adaptation lightweight. The `lora_alpha=64` scales the low-rank updates, allowing for flexibility in adjusting the magnitude of the adaptations. The `lora_dropout=0.05` adds some regularization by dropping a fraction of the low-rank connections during training, reducing the risk of overfitting. The `bias="none"` setting means that the bias terms in the target modules are not adapted, which can sometimes help reduce unnecessary complexity.

By calling `get_peft_model(model, config)` I integrate the LoRA modifications into the model, making it more efficient while retaining the benefits of large pre-trained models. Finally, I use `print_trainable_parameters(model)` to print the model's trainable parameters, helping me keep track of how much of the model has been adapted using LoRA. This technique is especially beneficial when dealing with very large models, such as transformer-based architectures, where full fine-tuning would otherwise be prohibitively expensive in terms of both memory and computation.

```py
if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

model = accelerator.prepare_model(model)
```
Here i am checking if my system has more than one GPU by using `torch.cuda.device_count()`. If the condition is true I set the `model.is_parallelizable` flag to True which indicates that the model can be parallelized across multiple devices. Additionally I enable `model.model_parallel` allowing the model to split its computations across the GPUs ensuring that the training process can leverage the power of multiple GPUs to accelerate the model's performance. This setup is especially useful for training large models that can't fit entirely on a single GPU, allowing them to be distributed efficiently across multiple devices.

```py
import transformers
from datetime import datetime
import gc

project = "journal-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=25,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_steps=1500,
        learning_rate=5e-6, # Want a small lr for finetuning
        num_train_epochs=30,
        weight_decay=0.01,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,            
        logging_dir="./logs",        
        save_strategy="steps",       
        save_steps=25,                
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        evaluation_strategy="steps", 
        eval_steps=25,               
        do_eval=True,                
        report_to="wandb",           
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  

trainer.train()
```
In this code I am using the **Hugging Face Transformers** library to fine-tune the **Mistral** model. The **Trainer** is initialized with the pre-trained model, the tokenized training and validation datasets, and the necessary training arguments. Here’s an explanation of the functionality of each parameter.

#### Key Components

1. **Model and Dataset Setup**:
   - The model is set with the pre-trained `mistral` model.
   - The `train_dataset` and `eval_dataset` are the tokenized training and validation datasets.

2. **Training Arguments (`TrainingArguments`)**:
   - **`output_dir`**: The directory where the model checkpoints and logs will be stored. It is dynamically named based on the model and project (`"./mistral-journal-finetune"`).
   - **`warmup_steps=25`**: The number of steps during which the learning rate will be gradually increased. This prevents the model from making large updates at the beginning of training.
   - **`per_device_train_batch_size=16`**: The batch size that is processed on each device (e.g., GPU). A batch size of 16 is chosen here to ensure efficient training without overloading memory.
   - **`gradient_accumulation_steps=8`**: Instead of updating the model after each batch, gradients are accumulated over 8 steps, which effectively increases the batch size without requiring additional memory.
   - **`gradient_checkpointing=True`**: This helps save memory by not storing intermediate activations during the forward pass, which will be recomputed during the backward pass.
   - **`max_steps=1500`**: The total number of steps for training. This ensures that the training process doesn’t run indefinitely.
   - **`learning_rate=5e-6`**: A small learning rate is used for fine-tuning, as large models like Mistral require careful adjustments to avoid destabilizing training.
   - **`num_train_epochs=30`**: The model will be trained for 30 epochs. Given the use of a pre-trained model, fewer epochs may be sufficient.
   - **`weight_decay=0.01`**: This regularization technique prevents overfitting by penalizing large weights during training.
   - **`bf16=True`**: This enables **bfloat16 precision**, which speeds up training while using less memory, ideal for large models like Mistral.
   - **`optim="paged_adamw_8bit"`**: The optimizer used is **AdamW**, with **8-bit precision** to reduce memory usage during training.
   - **`logging_steps=25`**: Logs are reported every 25 steps. This includes training and evaluation metrics, allowing us to track progress.
   - **`logging_dir="./logs"`**: The directory where logs will be stored.
   - **`save_strategy="steps"`**: Checkpoints are saved every step, ensuring that we don't lose progress in case of interruptions.
   - **`save_steps=25`**: The model is saved every 25 steps, and the best model (based on evaluation loss) will be stored.
   - **`load_best_model_at_end=True`**: Ensures that the best model (based on the evaluation loss) is loaded at the end of training.
   - **`metric_for_best_model="eval_loss"`**: The metric used to determine the best model is the **evaluation loss**.
   - **`evaluation_strategy="steps"`**: The model will be evaluated at regular intervals based on the specified number of steps.
   - **`eval_steps=25`**: Evaluations will be performed every 25 steps, allowing us to track how well the model is performing during training.
   - **`do_eval=True`**: Enables evaluation during training.
   - **`report_to="wandb"`**: Logs the training process to **Weights and Biases** (W&B) for tracking experiments, metrics, and visualizations.
   - **`run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"`**: This dynamically names the W&B run, incorporating the timestamp to avoid overwriting previous runs.

3. **Data Collator (`DataCollatorForLanguageModeling`)**:
   - **`data_collator`**: This parameter specifies how the dataset will be prepared for the model. Here, **`DataCollatorForLanguageModeling`** is used, which prepares the data for a **language modeling task**. The `mlm=False` indicates that this task is not a masked language modeling task.

4. **Disable Model Caching**:
   - **`model.config.use_cache = False`**: Caching is disabled during training to avoid unnecessary memory usage. This is particularly important during fine-tuning, as the model is already pre-trained, and caching could lead to excessive memory consumption.

5. **Training the Model**:
   - Finally, the model is trained using the **`trainer.train()`** method, which will execute the training loop based on the configuration specified in the `TrainingArguments`.

It is clearly seen that the responses are much more accurate and the model has learned which indicates that the fine tune has been quite ok.

The responses are as follows 

```py

User: hello
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

Assistant: 👋 Hello, my creator is online
User: give me specification of tosibox 175

Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

Assistant: ✅ The [Tosibox 175](https://www.tosibox.com/products/network-devices/los) is a device with global coverage and two LTE modems that can operate in different regions. It also has integrated WiFi access point for connecting devices locally. Its web UI allows remote management, and it is designed to work in both static and dynamic IP environments.
User: what is its specifications

Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
 

Assistant: ✅ The Tosibox 175's technical specifications include dimensions of 94 x 62 x 30 mm (W x H x D), weight of 80 g, and an operating temperature range from -20 °C to +65 °C (-4 °F to 149 °F). It operates on the frequency bands FDD B1 / B3 / B5 / B8 / B20 / B28 at 1–2 GHz, as well as TDD B38 / B40 / B41 at 2–2.5 GHz. Modem speed is up to 150 Mbps downlink and
User: does it have wifi

Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

Assistant: ✅ Yes, the Tosibox 175 has built-in WLAN access point supporting IEEE 802.11 b/g/n standards, with speeds up to 150 Mbps. This allows for wireless connections within range, which can be helpful for nearby users or IoT devices.
User: does it have usb

Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

Assistant: ✅ No, the Tosibox 175 does not have any USB ports. It connects via Ethernet only. However, if you need to interface with other devices, an analog modbus server may help.

```







