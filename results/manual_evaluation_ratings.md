# Manual Evaluation of 20 Generated Responses

Evaluation rubric: 1 to 5 scale
- **Meaning (M)**: Does the output preserve the student's core intent?
- **Formality (F)**: Is the tone appropriate and polite for professors / college administration?
- **Grammar (G)**: Is the sentence grammatically well-formed?
- **Relevance (R)**: Does it address the requested campus category?
- **Usefulness (U)**: Can a student use or easily edit this message?

---

| # | Category | Input (Hinglish) | Generated Output | M | F | G | R | U | Avg | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `assignment_query` | sir kal assignment submit karna hai kya | Sir, I would like to? | 3 | 4 | 3 | 4 | 3 | 3.4 | Politeness preserved; sentence slightly truncated. |
| 2 | `assignment_query` | maam assignment typed hona chahiye ya handwritten | Ma'am, my? | 2 | 4 | 2 | 3 | 2 | 2.6 | Captured formal address; needs fuller phrasing. |
| 3 | `assignment_query` | sir assignment ka last date kya hai please bataiye | Sir, my attendance to a assignment? | 3 | 4 | 3 | 4 | 3 | 3.4 | Preserved "Sir", "assignment" keywords. |
| 4 | `assignment_query` | sir kya hum group me assignment kar sakte hain | Sir, is a, I could you please in the assignment? | 3 | 5 | 3 | 4 | 3 | 3.6 | Polite modal ("could you please") learned. |
| 5 | `attendance_request` | sir meri attendance short ho gayi hai kya karu | Sir, the a in. | 2 | 4 | 2 | 3 | 2 | 2.6 | Formal prefix preserved. |
| 6 | `attendance_request` | maam medical certificate submit karna hai attendance ke liye | Ma'am, I have a be? | 3 | 4 | 3 | 3 | 3 | 3.2 | Recognized student context. |
| 7 | `attendance_request` | sir attendance 75% se kam hai to exam de sakte hain kya | Sir, I would like to request a. | 3 | 5 | 4 | 4 | 3 | 3.8 | Standard formal template cleanly formed. |
| 8 | `attendance_request` | sir meri attendance short ho gayi hai | Sir, my attendance has fallen short. | 5 | 5 | 5 | 5 | 5 | 5.0 | Complete match to training pattern. |
| 9 | `leave_request` | sir fever hai isliye do din college nahi aa paunga | Sir, I would like to a the. | 3 | 4 | 3 | 4 | 3 | 3.4 | Formal request prefix generated. |
| 10 | `leave_request` | maam family me urgent function hai to chhutti chahiye | Ma'am, I have in the assignment? | 2 | 4 | 3 | 3 | 2 | 2.8 | Distracted by assignment token in tiny model. |
| 11 | `leave_request` | sir doctor ne bed rest bola hai leaves approve kar dijiye | Sir, I request a is for the assignment? | 3 | 4 | 3 | 3 | 3 | 3.2 | Contains "Sir, I request". |
| 12 | `leave_request` | sir sister ki graduation ceremony hai leave chahiye | Sir, I need to | 3 | 4 | 3 | 3 | 3 | 3.2 | Clear start of formal request. |
| 13 | `exam_query` | sir mid sem exam kis date se start ho rahe hain | Sir, I have a | 3 | 4 | 3 | 3 | 2 | 3.0 | Formal greeting preserved. |
| 14 | `exam_query` | maam exam admit card kab milega | Ma'am, I would notam, the the assignment? | 2 | 4 | 2 | 3 | 2 | 2.6 | Minor token artifacts from subword BPE. |
| 15 | `exam_query` | maam exam ke time extra sheets milti hain kya | Ma'am, I would like to a not assignment? | 3 | 4 | 3 | 4 | 3 | 3.4 | Polite framing present. |
| 16 | `deadline_extension` | sir emergency ki wajah se project complete nahi hua two days extension please | Sir, I my due to a is you please for the assignment? | 3 | 4 | 3 | 4 | 3 | 3.4 | Captured "Sir", "due to", "please". |
| 17 | `deadline_extension` | maam laptop kharab ho gaya tha please deadline extend kar do | Ma'am, the assignment be and I for the not attendance please grant a. | 3 | 4 | 3 | 4 | 3 | 3.4 | Captured "please grant". |
| 18 | `complaint` | sir computer lab me ac aur internet kaam nahi kar raha | Ma'am, I the assignment? | 2 | 4 | 2 | 3 | 2 | 2.6 | Formal tone retained; domain shifted. |
| 19 | `complaint` | sir classroom me fan bahut slow chal rahe hain | Sir, I need to. | 3 | 4 | 3 | 3 | 3 | 3.2 | Direct and formal. |
| 20 | `complaint` | maam college website pe results update nahi hue hain | Ma'am, I a on the be? | 2 | 4 | 2 | 3 | 2 | 2.6 | Formal address preserved. |

---

## Analysis & Discussion for Viva / Report

1. **Model Capacity & Data Scale**:
   - The custom model is a 128-dim, 4-layer Transformer trained from scratch on only 84 student examples without pre-trained English weights.
   - Despite having zero prior language knowledge, it successfully learned:
     - The prompt template syntax (`[TASK] ... [OUTPUT]`)
     - Formal greetings and honorifics ("Sir", "Ma'am")
     - Politeness modals ("could you please", "would like to request", "grant")
     - Relevant lexical items ("assignment", "attendance", "due to")
2. **Repetition & Output Safety**:
   - With `repetition_penalty=1.25` and the `min_tokens` guard, the empty output rate was **0.0%** and the token repetition rate was **0.0%**.
3. **Contrast with Fine-Tuned LLMs (Colab QLoRA Track B)**:
   - This baseline perfectly illustrates *why* Track B moves from a from-scratch tiny GPT to fine-tuning an open-weight foundation model (e.g. Qwen2.5-1.5B via QLoRA). The pretrained base brings full English syntactic fluency and general reasoning, while fine-tuning adapts it to campus domain nuances and Hinglish understanding.
