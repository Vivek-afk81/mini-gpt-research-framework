"""
LexiFlow — Main orchestration pipeline.

Ties together: conversation memory → RAG retrieval → LLM generation → response.
Supports both the from-scratch baseline GPT and a fine-tuned HuggingFace model.
"""

import torch
from src.memory.chat_memory import ChatMemory
from src.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LexiFlow, a helpful campus assistant for college students.

Your responsibilities:
- Answer questions about college policies, attendance, examinations, assignments, leave, and campus facilities.
- Help students write formal messages to professors and administration.
- Use information from campus documents when available.
- Be polite, clear, and concise.
- If you cite a policy, mention the source document.
- If you don't know the answer, say so honestly.

You understand both Hinglish (Romanized Hindi-English mix) and English.
Always respond in clear, formal English."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LexiFlowPipeline:
    """
    Main pipeline that orchestrates memory, retrieval, and generation.

    Supports two backends:
        - "baseline": From-scratch GPT (for comparison)
        - "finetuned": QLoRA fine-tuned HuggingFace model
    """

    def __init__(self, backend="finetuned", model_name=None, device=None):
        self.backend = backend
        self.memory = ChatMemory(max_turns=5)
        self.retriever = Retriever()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model components (lazy-loaded)
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def load_model(self):
        """Load the appropriate model based on backend setting."""
        if self.backend == "finetuned":
            self._load_finetuned_model()
        else:
            self._load_baseline_model()

    def _load_finetuned_model(self):
        """Load the QLoRA fine-tuned HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_name = self._model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        adapter_path = "checkpoints/qlora"

        print(f"Loading base model: {base_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Try loading with adapter, fall back to base model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            self._model = PeftModel.from_pretrained(base_model, adapter_path)
            print(f"Loaded fine-tuned model with LoRA adapter from {adapter_path}")
        except Exception:
            print(f"No LoRA adapter found at {adapter_path}. Using base model.")
            self._model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()

    def _load_baseline_model(self):
        """Load the from-scratch GPT baseline."""
        from src.model.gpt import GPT
        from src.tokenizer import GPT2Tokenizer

        self._tokenizer = GPT2Tokenizer()

        checkpoint = torch.load(
            "checkpoints/best_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint["config"]

        self._model = GPT(
            vocab_size=self._tokenizer.vocab_size,
            d_model=config.model.d_model,
            context_length=config.model.context_length,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
        ).to(self.device)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        print("Loaded baseline GPT model")

    def load_knowledge_base(self):
        """Load or build the FAISS index from campus documents."""
        self.retriever.load_index()

    def chat(self, user_message, top_k=3, max_new_tokens=256, temperature=0.7):
        """
        Process a user message through the full pipeline:
        memory → retrieval → generation → response.

        Returns:
            dict with keys: response, sources, history
        """
        # 1. Retrieve relevant documents
        results = self.retriever.retrieve(user_message, top_k=top_k)
        context = self.retriever.format_context(results)
        sources = [
            {"title": chunk.title, "source": chunk.source, "score": score}
            for chunk, score in results
        ]

        # 2. Add user message to memory
        self.memory.add_user_message(user_message)

        # 3. Generate response
        if self.backend == "finetuned":
            response = self._generate_finetuned(
                user_message, context, max_new_tokens, temperature,
            )
        else:
            response = self._generate_baseline(user_message, context)

        # 4. Add response to memory
        self.memory.add_assistant_message(response)

        return {
            "response": response,
            "sources": sources,
            "history": self.memory.format_as_chat_messages(),
        }

    def _generate_finetuned(self, user_message, context, max_new_tokens, temperature):
        """Generate a response using the fine-tuned HuggingFace model."""
        # Build messages in chat format
        history = self.memory.format_for_prompt()

        prompt = f"""{SYSTEM_PROMPT}

## Campus Documents
{context}

## Conversation History
{history}

Please answer the student's latest question based on the campus documents and conversation context. If the answer is in the documents, cite the source."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        # Tokenize with chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = f"{prompt}\n\nUser: {user_message}\nAssistant:"

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def _generate_baseline(self, user_message, context):
        """Generate using the from-scratch baseline GPT (simple prompt completion)."""
        from src.config import Config
        config = Config()

        prompt = (
            f"{config.task_tag} campus_query "
            f"{config.tone_tag} formal "
            f"{config.input_tag} {user_message} "
            f"{config.output_tag}"
        )

        input_ids = self._tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)

        generated = list(input_ids)

        with torch.no_grad():
            for _ in range(64):
                seq = torch.tensor([generated[-128:]], device=self.device)
                logits = self._model(seq)
                next_logit = logits[0, -1, :]
                next_token = torch.argmax(next_logit).item()

                if next_token == self._tokenizer.eos_token_id:
                    break
                generated.append(next_token)

        output_ids = generated[len(input_ids):]
        return self._tokenizer.decode(output_ids).strip()

    def clear_memory(self):
        """Reset conversation history."""
        self.memory.clear()
