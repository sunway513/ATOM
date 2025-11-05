import asyncio
import itertools
import logging
import time
from dataclasses import fields
from typing import List, Union

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from atom.config import Config
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

logger = logging.getLogger("atom")


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id

        self.rquest_ids = set()
        self.io_processor = InputOutputProcessor(
            self.tokenizer, config.kv_cache_block_size
        )
        self.core_mgr = CoreManager(config)
        self._step_lock = None
        self._pending_results = {}
        logger.info("LLMEngine init")

    def add_request(
        self, 
        prompt_or_tokens_list: List[Union[str, List[int]]], 
        sampling_params_list: SamplingParams | List[SamplingParams]
    ):
        # if sampling params is not list, use it for all prompts
        if not isinstance(sampling_params_list, list):
            sampling_params_iter = itertools.repeat(sampling_params_list)
        else:
            # otherwise check num elements first
            if len(prompt_or_tokens_list) != len(sampling_params_list):
                raise ValueError(
                    f"number of elements in prompt_or_tokens_list and sampling_params_list is different: "
                    f"{len(prompt_or_tokens_list)=} vs {len(sampling_params_list)=}"
                )
            sampling_params_iter = sampling_params_list
        
        reqs = []
        for prompt, sampling_param in zip(prompt_or_tokens_list, sampling_params_iter):
            req = self.io_processor.preprocess(prompt, sampling_param)
            reqs.append(req)
        self.core_mgr.add_request(reqs)

    def step(self) -> list[Sequence]:
        seqs = self.core_mgr.get_output()
        return seqs

    def is_finished(self):
        return not self.io_processor.has_pending_requests()

    def generate(
        self,
        # prompts: list[str] | list[list[int]],
        prompts: list[str],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[str]:
        self.add_request(prompts, sampling_params)
        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        return outputs

    async def generate_async(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ):
        # Initialize lock on first use
        if self._step_lock is None:
            self._step_lock = asyncio.Lock()
        
        # Add the request and get its sequence ID
        req = self.io_processor.preprocess(prompt, sampling_params)
        seq_id = req.id
        self.core_mgr.add_request([req])
        
        while True:
            # Check if result is already available
            if seq_id in self._pending_results:
                result = self._pending_results.pop(seq_id)
                yield result
                return
            
            if seq_id not in self.io_processor.requests:
                break
            
            if not (self.core_mgr.is_alive() or self.core_mgr.is_rest()):
                break
            
            # Coordinate step() calls across all concurrent requests
            async with self._step_lock:
                seqs = self.step()
                outs = self.io_processor.postprocess(seqs)
                
                # Store all results so other waiting tasks can find them
                self._pending_results.update(outs)
            
            if seq_id in self._pending_results:
                result = self._pending_results.pop(seq_id)
                yield result
                return
            
            # Let other tasks run before trying again
            await asyncio.sleep(0)
    
    def start_profile(self):
        self.core_mgr.send_utility_command("start_profile")
        logger.info("Profiling started")
    
    def stop_profile(self):
        self.core_mgr.send_utility_command("stop_profile")
        logger.info("Profiling stopped. Trace files should be generated.")


class InputOutputProcessor:

    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.requests = {}

    def preprocess(
        self, prompt_or_tokens: str | list[int], sampling_params: SamplingParams
    ):
        """responsible for:
        1) Tokenize
        2) Create Sequence object"""
        tokens = (
            self.tokenizer.encode(prompt_or_tokens)
            if isinstance(prompt_or_tokens, str)
            else prompt_or_tokens
        )
        
        stop_token_sequences = []
        if sampling_params.stop_strings:
            stops = [sampling_params.stop_strings] if isinstance(sampling_params.stop_strings, str) else sampling_params.stop_strings
            for stop_str in stops:
                # Encode the full stop string as a sequence of tokens
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_tokens:
                    stop_token_sequences.append(stop_tokens)
        
        seq = Sequence(tokens, self.block_size, sampling_params, stop_token_sequences)
        seq.arrive_time = time.time()
        self.requests[seq.id] = seq
        print(
            f"Request {seq.id} arrived, input tokens: {len(tokens)}, pending requests: {len(self.requests)}"
        )
        return seq

    def postprocess(self, reqs: List[Sequence]):
        """responsible for:
        1) Compute stats for logging
        2) Detokenize"""
        outputs = {}
        for req in reqs:
            self.requests.pop(req.id)
            output_str = self.tokenizer.decode(req.completion_token_ids)
            req.leave_time = time.time()
            
            # Calculate TTFT (Time To First Token) and TPOT (Time Per Output Token)
            ttft = 0.0
            tpot = 0.0
            if req.first_token_time > 0:
                ttft = req.first_token_time - req.arrive_time
                # Calculate TPOT only if there are multiple output tokens
                if req.num_completion_tokens > 1:
                    tpot = (req.leave_time - req.first_token_time) / (req.num_completion_tokens - 1)
            
            print(
                f"Request {req.id} finished with reason {req.leave_reason}. "
                f"Input tokens: {req.num_prompt_tokens}, output tokens: {req.num_completion_tokens}, "
                f"latency: {req.leave_time - req.arrive_time:.2f}s, "
                f"TTFT: {ttft:.3f}s, TPOT: {tpot:.3f}s"
            )
            outputs[req.id] = {
                "text": output_str,
                "token_ids": req.completion_token_ids,
                "latency": req.leave_time - req.arrive_time,
                "finish_reason": req.leave_reason,
                "num_tokens_input": req.num_prompt_tokens,
                "num_tokens_output": req.num_completion_tokens,
                "ttft": ttft,  # Time to first token in seconds
                "tpot": tpot,  # Time per output token in seconds
            }
        return outputs

    def has_pending_requests(self):
        return len(self.requests) > 0
