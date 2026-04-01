"""
Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled Model Wrapper.

This is the Tier 1 (coordinator/reasoner) in the three-tier architecture.
It handles task decomposition, structured reasoning, and final answer
synthesis. It runs on ~16.5GB VRAM with Q4_K_M quantization and has
a 262K token context window.

The model was fine-tuned on Claude 4.6 Opus reasoning traces, giving it
structured <think>...</think> chain-of-thought behavior:

    <think>
    Let me analyze this request carefully:
    1. Identify the core objective...
    2. Break the task into subcomponents...
    3. Evaluate constraints...
    </think>
    The answer is 42.

This file provides:

  parse_think_output(raw_text) -> ThinkExtraction
    Parses model output into thinking content and answer. Also extracts
    numbered reasoning steps for the think-Koopman bridge.

  extract_reasoning_embedding(thinking_text, tokenizer, model)
    Runs the <think> content through the model and extracts hidden states
    from the last 4 layers, mean-pooled to a fixed-size vector. This
    vector is what gets projected into Koopman slot 2 (reasoning state)
    by the ThinkKoopmanBridge (shared_memory/think_koopman_bridge.py).

  QwenCoordinator class
    Main wrapper with three modes of operation:
    - reason(query, context): structured reasoning with <think> extraction
    - decompose(goal, agent_roster): produce task DAG for TS orchestrator
    - tool_call(messages, tools): tool-calling conversation turn

    The decompose() method produces JSON task specs compatible with the
    TypeScript orchestrator's ParsedTaskSpec format (orchestrator.ts).
    Tool call extraction uses bracket-depth-matching JSON parsing to
    handle nested tool inputs.

Dependencies:
  - shared_memory/think_koopman_bridge.py consumes the reasoning embeddings
  - orchestration/__init__.py exposes this model via the ToolServer
  - pipeline.py wires this as the primary reasoner and parser specialist
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Generator

import numpy as np


# Think token extraction

@dataclass
class ThinkExtraction:
    """Parsed output from the Qwen3.5-Claude model."""
    thinking: str # Raw content inside <think>...</think>
    answer: str # Content after </think>
    thinking_tokens: int = 0 # Token count of thinking portion
    answer_tokens: int = 0 # Token count of answer portion
    thinking_steps: List[str] = field(default_factory=list) # Numbered steps
    latency_ms: float = 0.0


def parse_think_output(raw: str) -> ThinkExtraction:
    """
    Parse model output into thinking and answer components.

    The model produces: <think> {reasoning} </think>\n {answer}
    """
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        answer = raw[think_match.end():].strip()
    else:
        thinking = ""
        answer = raw.strip()

    # Extract numbered reasoning steps
    steps = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n*$)', thinking, re.DOTALL)
    steps = [s.strip() for s in steps if s.strip()]

    return ThinkExtraction(
        thinking=thinking,
        answer=answer,
        thinking_steps=steps,
    )


def extract_reasoning_embedding(
    thinking_text: str,
    tokenizer,
    model,
    layer_indices: List[int] = None,
) -> np.ndarray:
    """
    Extract a dense representation of the <think> content for Koopman slot 3.

    Takes the hidden states from designated layers during a forward pass
    over the thinking tokens, then mean-pools to produce a fixed-size vector.

    Args:
        thinking_text: Raw text from inside <think>...</think>
        tokenizer: Model tokenizer
        model: The Qwen model (for hidden state extraction)
        layer_indices: Which layers to extract from (default: last 4)

    Returns:
        reasoning_embedding: (d,) numpy array suitable for Koopman operator
    """
    import torch

    if not thinking_text:
        return np.zeros(model.config.hidden_size, dtype=np.float64)

    layer_indices = layer_indices or [-4, -3, -2, -1]

    inputs = tokenizer(
        thinking_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

    # Extract and pool hidden states from designated layers
    hidden_states = outputs.hidden_states # tuple of (B, L, d)
    num_layers = len(hidden_states)

    selected = []
    for idx in layer_indices:
        actual_idx = idx if idx >= 0 else num_layers + idx
        if 0 <= actual_idx < num_layers:
            # Mean pool over sequence length
            layer_repr = hidden_states[actual_idx].mean(dim=1) # (B, d)
            selected.append(layer_repr)

    if not selected:
        return np.zeros(model.config.hidden_size, dtype=np.float64)

    # Average across selected layers
    stacked = torch.stack(selected, dim=0).mean(dim=0) # (B, d)
    return stacked.squeeze(0).cpu().numpy().astype(np.float64)


# Qwen3.5 Coordinator Model

class QwenCoordinator:
    """
    Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled wrapper.

    Serves dual roles:
      1. Coordinator: decomposes goals into task DAGs for the TS orchestrator
      2. Reasoner: multi-step reasoning with <think> chain-of-thought

    The <think> tokens are extracted and projected into the Koopman shared
    memory (slot 3: reasoning state) so other agents can query the
    coordinator's reasoning trace without serializing it as text.

    Usage:
        coordinator = QwenCoordinator()

        # As reasoner
        result = coordinator.reason(query, context)
        print(result.answer)
        print(result.thinking_steps)

        # Extract reasoning state for Koopman slot 3
        embedding = coordinator.extract_reasoning_state(result.thinking)

        # As task decomposer for TS orchestrator
        tasks = coordinator.decompose(goal, agent_roster)
    """

    MODEL_NAME = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"

    def __init__(
        self,
        model_name: str = None,
        quantization: str = "auto",
        max_new_tokens: int = 4096,
        device_map: str = "auto",
        think_layer_indices: List[int] = None,
    ):
        """
        Args:
            model_name: HuggingFace model ID (default: the Claude-distilled model)
            quantization: "auto", "4bit", "8bit", or "none"
            max_new_tokens: Max generation length
            device_map: Device placement strategy
            think_layer_indices: Layers to extract for reasoning embedding
        """
        self.model_name = model_name or self.MODEL_NAME
        self.max_new_tokens = max_new_tokens
        self.think_layer_indices = think_layer_indices or [-4, -3, -2, -1]

        self.model = None
        self.tokenizer = None
        self._hidden_size = None
        self._quantization = quantization
        self._device_map = device_map

    def load(self):
        """Load the model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        load_kwargs = {
            "device_map": self._device_map,
            "trust_remote_code": True,
        }

        if self._quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self._quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
        elif self._quantization == "none":
            load_kwargs["torch_dtype"] = torch.float16
        else:
            # Auto: use float16 if GPU, float32 if CPU
            load_kwargs["torch_dtype"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self.model.eval()
        self._hidden_size = self.model.config.hidden_size

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"QwenCoordinator ready: {self.model_name}")
        print(f" Parameters: {total_params:,}")
        print(f" Hidden size: {self._hidden_size}")
        print(f" Max context: {getattr(self.model.config, 'max_position_embeddings', 'unknown')}")

    @property
    def hidden_size(self) -> int:
        return self._hidden_size or 3584 # Qwen3.5-27B default

    # Reasoning (Tier 1 primary function)

    def reason(
        self,
        query: str,
        context: str = "",
        system_prompt: str = None,
        temperature: float = 0.6,
        max_tokens: int = None,
    ) -> ThinkExtraction:
        """
        Run structured reasoning with <think> chain-of-thought.

        The model produces:
            <think>
            Let me analyze this request carefully:
            1. ...
            2. ...
            </think>
            {final answer}

        Args:
            query: User question
            context: Retrieved context from SKA retriever
            system_prompt: Override system prompt
            temperature: Sampling temperature (0.6 for reasoning)
            max_tokens: Override max_new_tokens

        Returns:
            ThinkExtraction with thinking, answer, and parsed steps
        """
        import torch

        if self.model is None:
            self.load()

        system = system_prompt or (
            "You are an expert reasoning assistant. Think step by step inside "
            "<think> tags before giving your final answer. Be precise and thorough."
        )

        messages = [{"role": "system", "content": system}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            })
        else:
            messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=262144, # 262K context
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        start = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        elapsed_ms = (time.time() - start) * 1000
        generated = outputs[0][input_len:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)

        result = parse_think_output(raw)
        result.latency_ms = elapsed_ms
        result.thinking_tokens = len(self.tokenizer.encode(result.thinking))
        result.answer_tokens = len(self.tokenizer.encode(result.answer))

        return result

    # Reasoning state extraction for Koopman slot 3

    def extract_reasoning_state(
        self,
        thinking_text: str,
    ) -> np.ndarray:
        """
        Extract a dense embedding of the reasoning trace for Koopman slot 3.

        Runs the <think> content through the model and extracts hidden states
        from the last 4 layers, mean-pooled to a fixed-size vector.

        This vector is written to SharedSpectralMemory so other agents
        (retriever, code executor) can query the coordinator's reasoning
        state through the Koopman operator without text serialization.

        Args:
            thinking_text: Content from ThinkExtraction.thinking

        Returns:
            (hidden_size,) numpy array
        """
        if self.model is None:
            self.load()

        return extract_reasoning_embedding(
            thinking_text,
            self.tokenizer,
            self.model,
            layer_indices=self.think_layer_indices,
        )

    # Task decomposition for TS orchestrator

    def decompose(
        self,
        goal: str,
        agent_roster: List[Dict[str, str]],
        max_tokens: int = 2048,
    ) -> List[Dict]:
        """
        Decompose a high-level goal into a task DAG for the TS orchestrator.

        Produces JSON-structured task specs compatible with the TS
        orchestrator's ParsedTaskSpec format:
            [{title, description, assignee, dependsOn}, ...]

        Args:
            goal: High-level objective
            agent_roster: List of {name, role, model} dicts describing agents
            max_tokens: Max generation tokens

        Returns:
            List of task spec dicts
        """
        import json

        roster_text = "\n".join(
            f"- **{a['name']}** ({a.get('model', 'unknown')}): {a.get('role', 'general')}"
            for a in agent_roster
        )

        system = (
            "You are a task coordinator. Decompose goals into concrete, actionable tasks "
            "and assign them to team members.\n\n"
            "## Team Roster\n"
            f"{roster_text}\n\n"
            "## Output Format\n"
            "Respond ONLY with a JSON array of task objects inside a ```json code fence.\n"
            "Each task: {\"title\": str, \"description\": str, \"assignee\": str, \"dependsOn\": [str]}"
        )

        result = self.reason(
            query=f"Decompose this goal into tasks:\n\n{goal}",
            system_prompt=system,
            temperature=0.3,
            max_tokens=max_tokens,
        )

        # Parse JSON from answer (may be in a code fence)
        answer = result.answer
        fence_match = re.search(r'```json\s*([\s\S]*?)```', answer)
        candidate = fence_match.group(1) if fence_match else answer

        array_start = candidate.find('[')
        array_end = candidate.rfind(']')
        if array_start == -1 or array_end == -1:
            # Fallback: check thinking for the JSON
            fence_match = re.search(r'```json\s*([\s\S]*?)```', result.thinking)
            if fence_match:
                candidate = fence_match.group(1)
                array_start = candidate.find('[')
                array_end = candidate.rfind(']')

        if array_start == -1 or array_end == -1:
            return [{"title": "Execute goal", "description": goal, "assignee": agent_roster[0]["name"], "dependsOn": []}]

        try:
            tasks = json.loads(candidate[array_start:array_end + 1])
            if not isinstance(tasks, list):
                raise ValueError("Not a list")
            return tasks
        except (json.JSONDecodeError, ValueError):
            return [{"title": "Execute goal", "description": goal, "assignee": agent_roster[0]["name"], "dependsOn": []}]

    # Tool-calling interface for TS orchestrator

    def tool_call(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict:
        """
        Run a tool-calling conversation turn.

        Compatible with the TS orchestrator's agent runner pattern.
        Returns the model's response with any tool_use blocks extracted.

        Args:
            messages: Conversation history in OpenAI/Anthropic format
            tools: Tool definitions (JSON schema format)
            temperature: Sampling temperature
            max_tokens: Max generation tokens

        Returns:
            Dict with 'content' (text blocks + tool_use blocks),
            'thinking' (extracted <think> content), 'usage' (token counts)
        """
        import torch

        if self.model is None:
            self.load()

        # Build prompt with tools in system message
        system_parts = []
        user_parts = []

        for msg in messages:
            if msg.get("role") == "system" or msg.get("role") == "developer":
                system_parts.append(msg["content"])
            elif msg.get("role") == "user":
                user_parts.append(msg["content"])

        if tools:
            tool_desc = "\n".join(
                f"- {t['name']}: {t.get('description', '')}" for t in tools
            )
            system_parts.append(
                f"\n## Available Tools\n{tool_desc}\n\n"
                "To use a tool, include a JSON block: "
                '{\"tool\": \"name\", \"input\": {...}}'
            )

        chat_messages = []
        if system_parts:
            chat_messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        for content in user_parts:
            chat_messages.append({"role": "user", "content": content})

        prompt = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=262144)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_len:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        result = parse_think_output(raw)

        # Extract tool calls from the answer using bracket-matching
        # (handles nested JSON unlike regex)
        tool_calls = []
        import json
        text = result.answer
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Find matching closing brace accounting for nesting
                depth = 0
                j = i
                while j < len(text):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and "tool" in obj:
                                    tool_calls.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                    j += 1
                i = j + 1
            else:
                i += 1

        return {
            "content": result.answer,
            "thinking": result.thinking,
            "thinking_steps": result.thinking_steps,
            "tool_calls": tool_calls,
            "usage": {
                "thinking_tokens": result.thinking_tokens,
                "answer_tokens": result.answer_tokens,
            },
        }
