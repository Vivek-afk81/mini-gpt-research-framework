"""
LexiFlow fine-tuning — Dataset formatter.

Converts the campus JSONL dataset into instruction-tuning format
compatible with HuggingFace chat templates (ChatML / Alpaca).
"""

import json
from pathlib import Path


def load_jsonl(path):
    """Load examples from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_for_instruction_tuning(examples):
    """
    Convert campus examples into instruction-tuning format.

    Each example becomes:
        system: You are LexiFlow, a campus assistant...
        user: <informal Hinglish message>
        assistant: <formal English response>
    """
    system_prompt = (
        "You are LexiFlow, a helpful campus assistant for college students. "
        "Convert the student's informal Hinglish message into a clear, "
        "polite, formal English message. Preserve the original meaning. "
        "The message category is: {category}."
    )

    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt.format(category=ex["category"].replace("_", " ")),
                },
                {
                    "role": "user",
                    "content": ex["input"],
                },
                {
                    "role": "assistant",
                    "content": ex["output"],
                },
            ]
        })

    return formatted


def format_for_qa_tuning(examples, knowledge_dir="data/knowledge"):
    """
    Create question-answering training examples from campus documents.

    Generates examples where the assistant answers campus policy questions
    with source citations.
    """
    qa_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "What is the minimum attendance required?"},
                {"role": "assistant", "content": "According to the Attendance Policy, all students must maintain a minimum attendance of 75% in each subject throughout the semester. If attendance falls below 75%, the student may be barred from the end-semester examination."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "sir assignment late submit kar sakte hain kya"},
                {"role": "assistant", "content": "According to the Assignment Guidelines, late submissions incur a penalty of 10% per day for up to 3 days. Assignments submitted more than 3 days late will not be accepted and will receive zero marks. I recommend submitting on time or requesting an extension from your instructor."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "maam medical leave ke liye kya process hai"},
                {"role": "assistant", "content": "According to the Leave Policy, for medical leave you must submit a medical certificate from a registered medical practitioner within 5 working days of resuming classes. The certificate should include the diagnosis, period of rest, and doctor's registration number. If the leave exceeds 5 consecutive days, approval from the Head of Department is required."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "exam me calculator le ja sakte hain kya"},
                {"role": "assistant", "content": "According to the Examination Rules, only non-programmable and non-graphing calculators are allowed during examinations, unless otherwise specified by the subject faculty. Mobile phones and smartwatches are strictly prohibited inside the examination hall."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "hostel me kab tak andar aa jana chahiye"},
                {"role": "assistant", "content": "According to the Hostel Rules, students must return to the hostel by 9:00 PM on weekdays and 10:00 PM on weekends. Late entry requires written permission from the hostel warden. If you need to stay out for more than 24 hours, you must fill out an outing form."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "plagiarism kitna allowed hai assignment me"},
                {"role": "assistant", "content": "According to the Assignment Guidelines, the maximum allowed similarity index is 20%. Assignments with similarity above 20% will be returned for revision. Assignments with similarity above 40% will receive zero marks and may result in disciplinary action."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "grace marks milte hain kya exam me"},
                {"role": "assistant", "content": "According to the Examination Rules, a provision of up to 5 grace marks per subject is available for students who narrowly fail. Grace marks are awarded at the discretion of the examination committee and are applicable only in End-Semester Examinations."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "library kab tak khuli rehti hai"},
                {"role": "assistant", "content": "According to the Hostel and Campus Rules, the central library is open from 8:00 AM to 9:00 PM on weekdays and 9:00 AM to 5:00 PM on weekends. During examination periods, library hours are extended to 11:00 PM."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "mid sem exam kab hai"},
                {"role": "assistant", "content": "According to the Academic Calendar, the Mid-Semester Examinations for the Odd Semester are scheduled from September 16 to September 21, and for the Even Semester from March 3 to March 8. Please check the college website for any updates to the schedule."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are LexiFlow, a campus assistant. Answer the student's question using campus policy documents. Cite the source."},
                {"role": "user", "content": "ragging report kaise karu"},
                {"role": "assistant", "content": "According to the Hostel and Campus Rules, the college has a zero-tolerance policy towards ragging. If you experience or witness ragging, please report it immediately to the anti-ragging committee, the hostel warden, or use the anonymous complaint hotline. Any student found involved in ragging will face immediate disciplinary action, which may include suspension or expulsion."},
            ]
        },
    ]

    return qa_examples


def save_chat_jsonl(data, output_path):
    """Save formatted chat data as JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {output_path}")


def prepare_finetuning_data(
    raw_path="data/raw/campus_messages.jsonl",
    output_dir="data/finetuning",
):
    """
    Prepare the complete fine-tuning dataset:
    - Instruction-tuning examples from Hinglish pairs
    - QA examples from campus documents
    """
    # Load and format Hinglish → English pairs
    raw_examples = load_jsonl(raw_path)
    instruction_data = format_for_instruction_tuning(raw_examples)

    # Add QA examples
    qa_data = format_for_qa_tuning(raw_examples)

    # Combine
    all_data = instruction_data + qa_data
    print(f"Total: {len(instruction_data)} instruction + {len(qa_data)} QA = {len(all_data)} examples")

    # Save
    save_chat_jsonl(all_data, f"{output_dir}/train_chat.jsonl")

    return all_data


if __name__ == "__main__":
    prepare_finetuning_data()
