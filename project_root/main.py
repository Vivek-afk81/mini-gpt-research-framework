import argparse
import os

from src.lexiflow.pipeline.orchestrator import LexiFlow


def main():
    parser = argparse.ArgumentParser(description="LexiFlow offline campus assistant")
    parser.add_argument("message", nargs="?", help="Hinglish student message")
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--name", default="")
    parser.add_argument("--roll-no", default="")
    parser.add_argument(
        "--backend", choices=("local", "groq"),
        default=os.getenv("TRANSLATOR_BACKEND", "local"),
        help="local uses the dataset fallback; groq uses Groq when GROQ_API_KEY is set",
    )
    args = parser.parse_args()

    message = args.message or input("Student message: ").strip()
    app = LexiFlow(backend=args.backend)
    if args.name or args.roll_no:
        app.memory.upsert_profile(args.user_id, name=args.name, roll_no=args.roll_no)
    result = app.run(args.user_id, message)
    print(f"\nEnglish: {result['english']}")
    print(f"Category: {result['category']} | Tone: {result['tone']}")
    print(f"\nDraft email:\n{result['draft']}")
    print("\nSources:")
    for index, source in enumerate(result["sources"], 1):
        print(f"[P{index}] {source['doc']} / {source['section']} ({source['score']:.2f})")


if __name__ == "__main__":
    main()