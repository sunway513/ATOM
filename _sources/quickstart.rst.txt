Quickstart
==========

This guide will get you started with ATOM in 5 minutes.

Serving a Model
---------------

.. code-block:: python

   from atom import LLM

   # Load model
   llm = LLM(
       model="meta-llama/Llama-2-7b-hf",
       gpu_memory_utilization=0.9,
       max_model_len=4096
   )

   # Generate text
   outputs = llm.generate("Hello, my name is", max_tokens=50)
   print(outputs[0].text)

Batch Inference
---------------

.. code-block:: python

   from atom import LLM

   llm = LLM(model="meta-llama/Llama-2-7b-hf")

   # Batch prompts
   prompts = [
       "The capital of France is",
       "The largest ocean is",
       "Python is a"
   ]

   # Generate in batch
   outputs = llm.generate(prompts, max_tokens=20)

   for output in outputs:
       print(f"Prompt: {output.prompt}")
       print(f"Output: {output.text}\n")

Distributed Serving
-------------------

Multi-GPU serving:

.. code-block:: python

   from atom import LLM

   # Use 4 GPUs with tensor parallelism
   llm = LLM(
       model="meta-llama/Llama-2-70b-hf",
       tensor_parallel_size=4,
       gpu_memory_utilization=0.95
   )

   outputs = llm.generate("Tell me about AMD GPUs", max_tokens=100)

API Server
----------

Start a RESTful API server:

.. code-block:: bash

   python -m atom.entrypoints.api_server \
       --model meta-llama/Llama-2-7b-hf \
       --host 0.0.0.0 \
       --port 8000

Query the server:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/generate",
       json={
           "prompt": "Hello, world!",
           "max_tokens": 50
       }
   )

   print(response.json()["text"])

Performance Tips
----------------

1. **GPU Memory**: Set `gpu_memory_utilization` to 0.9-0.95
2. **Batch Size**: Increase `max_num_batched_tokens` for throughput
3. **KV Cache**: Configure `block_size` based on workload
4. **Compilation**: Enable CUDAGraph for repeated inference

Next Steps
----------

* :doc:`architecture_guide` - Understand ATOM architecture
* :doc:`configuration_guide` - Configure for your workload
* :doc:`serving_benchmarking_guide` - Measure performance
