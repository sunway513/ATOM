Serving API
===========

LLM Class
---------

Main class for loading and serving models.

.. code-block:: python

   from atom import LLM

   llm = LLM(model="meta-llama/Llama-2-7b-hf")

**Parameters:**

* **model** (*str*) - HuggingFace model name or path
* **gpu_memory_utilization** (*float*) - GPU memory usage (0.0-1.0). Default: 0.9
* **max_model_len** (*int*) - Maximum sequence length
* **tensor_parallel_size** (*int*) - Number of GPUs for tensor parallelism. Default: 1
* **dtype** (*str*) - Model dtype ('float16', 'bfloat16', 'float32')

Methods
^^^^^^^

generate()
""""""""""

.. code-block:: python

   outputs = llm.generate(prompts, max_tokens=50)

Generate text from prompts.

**Parameters:**

* **prompts** (*str | list[str]*) - Input prompts
* **max_tokens** (*int*) - Maximum tokens to generate
* **temperature** (*float*) - Sampling temperature. Default: 1.0
* **top_p** (*float*) - Nucleus sampling threshold. Default: 1.0
* **top_k** (*int*) - Top-k sampling. Default: -1 (disabled)

**Returns:**

* **outputs** (*list[RequestOutput]*) - Generated outputs

SamplingParams
--------------

.. code-block:: python

   from atom import SamplingParams

   params = SamplingParams(
       temperature=0.8,
       top_p=0.95,
       max_tokens=100
   )

Configuration for text generation.

**Parameters:**

* **temperature** (*float*) - Controls randomness
* **top_p** (*float*) - Nucleus sampling threshold
* **top_k** (*int*) - Top-k sampling
* **max_tokens** (*int*) - Maximum tokens to generate
* **presence_penalty** (*float*) - Penalty for token presence
* **frequency_penalty** (*float*) - Penalty for token frequency

RequestOutput
-------------

Output from generation request.

**Attributes:**

* **prompt** (*str*) - Input prompt
* **text** (*str*) - Generated text
* **tokens** (*list[int]*) - Generated token IDs
* **finished** (*bool*) - Whether generation completed

Example
-------

Complete example:

.. code-block:: python

   from atom import LLM, SamplingParams

   # Initialize model
   llm = LLM(
       model="meta-llama/Llama-2-7b-hf",
       tensor_parallel_size=2,
       gpu_memory_utilization=0.9
   )

   # Configure sampling
   sampling_params = SamplingParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=200
   )

   # Generate
   prompts = ["Tell me about AMD GPUs"]
   outputs = llm.generate(prompts, sampling_params=sampling_params)

   for output in outputs:
       print(f"Prompt: {output.prompt}")
       print(f"Generated: {output.text}")
