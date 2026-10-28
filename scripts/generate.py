"""
LexiFlow text generation CLI.

Translates informal Romanized Hinglish student messages into formal English
using the custom decoder-only Transformer.

Usage:
    # Single generation
    python scripts/generate.py --input "sir kal assignment submit karna hai kya"

    # Batch test across categories
    python scripts/generate.py --batch-test

    # Interactive mode
    python scripts/generate.py --interactive
"""

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generation import TextGenerator, detect_category


SAMPLE_MESSAGES = [
    # 1. Assignment query
    ("sir kal assignment submit karna hai kya", "assignment_query"),
    ("maam assignment typed hona chahiye ya handwritten", "assignment_query"),
    ("sir assignment ka last date kya hai please bataiye", "assignment_query"),
    # 2. Attendance request
    ("sir meri attendance short ho gayi hai kya karu", "attendance_request"),
    ("maam medical certificate submit karna hai attendance ke liye", "attendance_request"),
    ("sir attendance 75% se kam hai to exam de sakte hain kya", "attendance_request"),
    # 3. Leave request
    ("sir fever hai isliye do din college nahi aa paunga", "leave_request"),
    ("maam family me urgent function hai to chhutti chahiye", "leave_request"),
    ("sir doctor ne bed rest bola hai leaves approve kar dijiye", "leave_request"),
    # 4. Exam query
    ("sir mid sem exam kis date se start ho rahe hain", "exam_query"),
    ("maam exam admit card kab milega", "exam_query"),
    ("sir kal ka exam kis classroom me conduct hoga", "exam_query"),
    # 5. Deadline extension
    ("sir tabiyat kharab thi assignment submit karne ke liye thoda extra time de dijiye", "deadline_extension"),
    ("maam laptop kharab ho gaya tha please deadline extend kar do", "deadline_extension"),
    ("sir emergency ki wajah se project complete nahi hua two days extension please", "deadline_extension"),
    # 6. Complaint / Admin
    ("sir computer lab me ac aur internet kaam nahi kar raha", "complaint"),
    ("maam classroom me projector switch on nahi ho raha", "complaint"),
    ("sir library me books issue nahi ho rahi system down hai", "complaint"),
]


def run_batch_test(generator: TextGenerator, temperature: float = 0.7):
    """Run generation on 18 diverse test messages across all categories."""
    print("\n" + "=" * 80)
    print("  LEXIFLOW BATCH INFERENCE TEST (CPU)")
    print("=" * 80)

    total_time = 0.0
    for idx, (msg, expected_cat) in enumerate(SAMPLE_MESSAGES, 1):
        t0 = time.time()
        result = generator.generate(
            input_text=msg,
            temperature=temperature,
            repetition_penalty=1.25,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        detected = result["category"]
        match_tag = "[OK]" if detected == expected_cat else f"[{detected}]"

        print(f"\n[{idx:02d}/18] Category: {expected_cat:<20} {match_tag} ({elapsed*1000:.1f}ms)")
        print(f"  Input : {msg}")
        print(f"  Output: {result['output']}")

    avg_time = (total_time / len(SAMPLE_MESSAGES)) * 1000
    print("\n" + "=" * 80)
    print(f"  Batch test completed in {total_time:.2f}s (Average {avg_time:.1f}ms per query on CPU)")
    print("=" * 80 + "\n")


def run_interactive(generator: TextGenerator, temperature: float = 0.7):
    """Run an interactive CLI session."""
    print("\n" + "=" * 65)
    print("  LexiFlow Interactive CLI — Hinglish to Formal English")
    print("  Type your Hinglish message and press Enter.")
    print("  Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("\n[Hinglish] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting LexiFlow. Goodbye!")
                break

            result = generator.generate(
                input_text=user_input,
                temperature=temperature,
                repetition_penalty=1.25,
            )

            print(f"[Category] : {result['category']}")
            print(f"[Formal]   : {result['output']}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LexiFlow — Command-line inference for Hinglish to Formal English conversion"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Hinglish input message to convert into formal English"
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        help="Message category (assignment_query, attendance_request, leave_request, exam_query, deadline_extension, complaint)"
    )
    parser.add_argument(
        "--tone", "-t", type=str, default="formal",
        help="Tone (default: formal)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (0 for greedy, >0 for sampling)"
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling cutoff (default: 40)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p / nucleus sampling cutoff (default: 0.9)"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.25,
        help="Repetition penalty factor (default: 1.25)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=48,
        help="Maximum generated tokens (default: 48)"
    )
    parser.add_argument(
        "--batch-test", action="store_true",
        help="Run inference on 18 sample messages across categories"
    )
    parser.add_argument(
        "--interactive", "-m", action="store_true",
        help="Launch interactive conversation shell"
    )

    args = parser.parse_args()

    # Load generator
    try:
        generator = TextGenerator(checkpoint_path=args.checkpoint)
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    if args.batch_test:
        run_batch_test(generator, temperature=args.temperature)
    elif args.input:
        try:
            result = generator.generate(
                input_text=args.input,
                category=args.category,
                tone=args.tone,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            print("\n" + "-" * 50)
            print(f"Input    : {result['input']}")
            print(f"Category : {result['category']}")
            print(f"Output   : {result['output']}")
            print("-" * 50 + "\n")
        except ValueError as err:
            print(f"\n[Input Error] {err}\n")
            sys.exit(1)
    else:
        # Default to interactive if no input is provided
        run_interactive(generator, temperature=args.temperature)


if __name__ == "__main__":
    main()
