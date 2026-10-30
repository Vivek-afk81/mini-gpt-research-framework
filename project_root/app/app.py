"""Gradio entry point for the LexiFlow demo."""

import sys
from pathlib import Path

import gradio as gr

# Allow `python app/app.py` to resolve the project package from project_root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lexiflow.pipeline.orchestrator import LexiFlow


pipeline = LexiFlow()


def process(user_id, name, roll_no, message):
    if not message or not message.strip():
        return "", "", "Please enter a message.", ""
    if name or roll_no:
        pipeline.memory.upsert_profile(user_id or "demo_user", name=name, roll_no=roll_no)
    result = pipeline.run(user_id or "demo_user", message)
    sources = "\n".join(
        f"[P{i + 1}] {hit['doc']} / {hit['section']} (score {hit['score']:.2f})"
        for i, hit in enumerate(result["sources"])
    ) or "No matching policy found."
    return result["english"], f"{result['category']} | {result['tone']}", result["draft"], sources


with gr.Blocks(title="LexiFlow") as demo:
    gr.Markdown("# LexiFlow\nHinglish to formal English, grounded in sample college policies.")
    with gr.Row():
        user_id = gr.Textbox(label="Demo user ID", value="demo_user")
        name = gr.Textbox(label="Name (optional)")
        roll_no = gr.Textbox(label="Roll number (optional)")
    message = gr.Textbox(label="Student message", lines=3, placeholder="meri attendance 67% hai, kya mujhe condonation mil sakta hai")
    button = gr.Button("Generate")
    english = gr.Textbox(label="Formal English")
    label = gr.Textbox(label="Category | Tone")
    email = gr.Textbox(label="Draft email", lines=8)
    sources = gr.Textbox(label="Policy sources", lines=4)
    button.click(process, [user_id, name, roll_no, message], [english, label, email, sources])


if __name__ == "__main__":
    demo.launch()