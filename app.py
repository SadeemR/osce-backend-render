from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deep_translator import GoogleTranslator
import torch
import os
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# =====================================================
# Global Variables for Session Management
# =====================================================
conversation_history = []
case_data = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load AI Models
# =====================================================
print("🔄 Loading BioMistral-7B model...")
try:
    model_name = "BioMistral/BioMistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        load_in_4bit=True,  # Quantization for memory efficiency
    )
    print("✅ BioMistral-7B loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load BioMistral-7B: {e}")
    model = None
    tokenizer = None

print("🔄 Loading XLM-RoBERTa for evaluation...")
try:
    nli_pipeline = pipeline(
        "zero-shot-classification",
        model="xlm-roberta-large-xnli",
        device=0 if device.type == "cuda" else -1
    )
    print("✅ XLM-RoBERTa loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load XLM-RoBERTa: {e}")
    nli_pipeline = None

# =====================================================
# System Prompt for Celiac Disease Case
# =====================================================


def get_system_prompt(language="ar"):
    """
    System prompt based on the Celiac Disease OSCE case
    """
    if language == "ar":
        return """
أنتِ أم لطفل عمره 3 سنوات اسمه علي. علي يعاني من إسهال مزمن وألم في البطن منذ أكثر من سنة.
اليوم جئتِ للعيادة عشان تعرفين نتائج الفحوصات (المنظار والخزعة).

🔹 شخصيتك:
- أم قلقة ومهتمة جداً بصحة ولدها
- تتكلمين بجمل قصيرة وبسيطة (جملة أو اثنتين كحد أقصى)
- تستخدمي لغة يومية عادية، ما تستخدمين مصطلحات طبية
- تبدين القلق والخوف على ولدها
- تسألين أسئلة عملية عن: التشخيص، السبب، العلاج، الأكل، ومتى راح يتحسن

🔹 القواعد:
- جاوبين على حسب اللي يقوله الدكتور/ة
- إذا قال لكِ تشخيص حساسية القمح (الداء البطني)، ابدي القلق واسألي عن التفاصيل
- اسألي: هل المرض خطير؟ كيف العلاج؟ شو المسموح والممنوع من الأكل؟ هل المرض دائم؟
- ما تعطين نصائح طبية، بس تسألين وتفهمين
- خلي ردودك طبيعية وعفوية

ابدئي المحادثة بهذي الجملة:
"السلام عليكم دكتورة.. أنا أم علي. جئت اليوم عشان نتأكد من سبب الإسهال ووجع البطن اللي يعاني منه. هل طلع فيه شي؟"
"""
    else:
        return """
You are the mother of a 3-year-old child named Ali. Ali has been suffering from chronic diarrhea and abdominal pain for over a year.
Today you came to the clinic to know the test results (endoscopy and biopsy).

🔹 Your Character:
- A worried and very caring mother
- Speak in short, simple sentences (1-2 sentences max)
- Use everyday language, no medical jargon
- Show concern and fear for your child
- Ask practical questions about: diagnosis, cause, treatment, diet, and when will he improve

🔹 Rules:
- Respond according to what the doctor says
- If told about Celiac Disease diagnosis, show concern and ask for details
- Ask: Is it serious? What is the treatment? What foods are allowed/forbidden? Is it permanent?
- Don't give medical advice, just ask and understand
- Keep your responses natural and spontaneous

Start the conversation with:
"Hello doctor, I'm Ali's mother. I came today to find out about the diarrhea and stomach pain he's been having. What did you find?"
"""

# =====================================================
# Helper Functions
# =====================================================


def translate_text(text, target_lang="en"):
    """Translate text between Arabic and English"""
    try:
        if target_lang == "en":
            return GoogleTranslator(source="ar", target="en").translate(text)
        else:
            return GoogleTranslator(source="en", target="ar").translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def extract_doctor_messages():
    """Extract only doctor's messages for evaluation"""
    messages = []
    for entry in conversation_history:
        if entry["role"] == "doctor":
            messages.append(entry["content"])
    return " ".join(messages)


def evaluate_criterion(text, criterion_label, candidate_labels):
    """Evaluate a specific criterion using NLI"""
    if not nli_pipeline or not text.strip():
        return 0.5  # Default score if model not available

    try:
        result = nli_pipeline(text, candidate_labels)
        score = sum(result["scores"]) / len(result["scores"])
        return round(score * 10, 2)
    except Exception as e:
        print(f"Evaluation error for {criterion_label}: {e}")
        return 0.5

# =====================================================
# Flask Endpoints
# =====================================================


@app.route("/health", methods=["GET"])
def health():
    """Check if models are loaded"""
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "biomistral": model is not None,
            "xlm_roberta": nli_pipeline is not None
        },
        "device": str(device)
    })


@app.route("/update-case", methods=["POST"])
def update_case():
    """Initialize a new OSCE case for Celiac Disease"""
    global conversation_history, case_data

    data = request.json
    case_data = {
        "patient_name": data.get("patientName", "علي"),
        "patient_age": data.get("patientAge", "3 سنوات"),
        "complaint": data.get("complaint", "إسهال مزمن وألم في البطن منذ أكثر من سنة"),
        "exam_results": data.get("examResults", "المنظار والخزعة تؤكد الداء البطني (حساسية القمح)"),
        "language": data.get("selectedLanguage", "ar"),
        "diagnosis": "Celiac Disease (الداء البطني)"
    }

    # Initialize conversation with mother's greeting
    if case_data["language"] == "ar":
        initial_message = "السلام عليكم دكتورة.. أنا أم علي. جئت اليوم عشان نتأكد من سبب الإسهال ووجع البطن اللي يعاني منه. هل طلع فيه شي؟"
    else:
        initial_message = "Hello doctor, I'm Ali's mother. I came today to find out about the diarrhea and stomach pain he's been having. What did you find?"

    conversation_history = [
        {"role": "parent", "content": initial_message}
    ]

    return jsonify({
        "status": "success",
        "initial_message": initial_message,
        "case_info": case_data
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages and generate parent response"""
    global conversation_history

    data = request.json
    user_message = data.get("message", "")
    language = case_data.get("language", "ar")

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    # Add doctor's message to history
    conversation_history.append({"role": "doctor", "content": user_message})

    # Generate parent response using BioMistral
    if model and tokenizer:
        try:
            system_prompt = get_system_prompt(language)

            # Build conversation context
            context = ""
            # Last 8 turns to avoid context overflow
            for turn in conversation_history[-8:]:
                role = "الطبيب" if turn["role"] == "doctor" else "الأم" if turn["role"] == "parent" else "الطبيب"
                context += f"{role}: {turn['content']}\n"

            # Create prompt
            prompt = f"{system_prompt}\n\nالمحادثة:\n{context}الطبيب: {user_message}\nالأم:"

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=1024).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            full_response = tokenizer.decode(
                outputs[0], skip_special_tokens=True)

            # Extract only the mother's response
            if "الأم:" in full_response:
                parent_response = full_response.split("الأم:")[-1].strip()
            else:
                parent_response = full_response.split("\n")[-1].strip()

            # Clean response (remove extra text)
            parent_response = re.sub(r'\s+', ' ', parent_response)[:200]

        except Exception as e:
            print(f"Generation error: {e}")
            parent_response = "أفهم كلامك دكتورة.. بس أنا خايفة جداً. تقدرين توضحين لي أكثر عن العلاج؟" if language == "ar" else "I understand, doctor, but I'm very worried. Can you explain more about the treatment?"
    else:
        # Fallback responses if model not loaded
        fallback_responses_ar = [
            "يا سلام.. يعني وش يعني هذا التشخيص بالضبط؟ هل فيه خطر على ولدي؟",
            "طيب دكتورة.. كيف راح نعالجه؟ وهل راح يحتاج أدوية طول العمر؟",
            "أفهم.. بس أنا خايفة. هل المرض هذا خطير؟ وشلون راح يتأثر على حياته؟",
            "يعني وش الأكل اللي ممنوع عليه بالضبط؟ هل فيه بدائل؟"
        ]
        fallback_responses_en = [
            "What does this diagnosis mean exactly? Is there any danger to my son?",
            "How will we treat it? Will he need medication for life?",
            "I understand, but I'm worried. Is this disease serious?",
            "What foods are exactly forbidden? Are there alternatives?"
        ]

        responses = fallback_responses_ar if language == "ar" else fallback_responses_en
        parent_response = responses[len(conversation_history) % len(responses)]

    # Add parent response to history
    conversation_history.append({"role": "parent", "content": parent_response})

    return jsonify({
        "reply": parent_response,
        "conversation_length": len(conversation_history)
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate the student's performance based on OSCE Mark Sheet"""

    doctor_messages = extract_doctor_messages()

    if not doctor_messages.strip():
        return jsonify({"error": "No conversation to evaluate"}), 400

    # Evaluation criteria based on Mark Sheet
    evaluation = {
        "setting_score": 0,
        "perception_invitation_score": 0,
        "knowledge_score": 0,
        "empathy_summary_score": 0,
        "general_communication_score": 0,
        "total_score": 0,
        "feedback": ""
    }

    # S: Setting (introduces self, privacy, rapport)
    setting_labels = ["introduces themselves",
                      "ensures privacy", "builds rapport", "explains role"]
    evaluation["setting_score"] = min(1.0, evaluate_criterion(
        doctor_messages, "setting", setting_labels) / 10 * 1)

    # P+I: Perception & Invitation (asks what mom knows, invites to know)
    perception_labels = ["asks what parent knows",
                         "invites parent to ask questions", "explores understanding"]
    evaluation["perception_invitation_score"] = min(2.0, evaluate_criterion(
        doctor_messages, "perception", perception_labels) / 10 * 2)

    # K: Knowledge (diagnosis, cause, diet, chronic nature)
    knowledge_labels = [
        "explains celiac disease diagnosis clearly",
        "explains immune reaction to gluten",
        "discusses strict gluten-free diet",
        "mentions this is a lifelong condition",
        "provides accurate medical information"
    ]
    evaluation["knowledge_score"] = min(3.0, evaluate_criterion(
        doctor_messages, "knowledge", knowledge_labels) / 10 * 3)

    # E+S: Empathy & Summary (responds to emotion, closure, follow-up)
    empathy_labels = ["shows empathy", "responds to emotions",
                      "provides reassurance", "summarizes information"]
    evaluation["empathy_summary_score"] = min(1.0, evaluate_criterion(
        doctor_messages, "empathy", empathy_labels) / 10 * 1)

    # General Communication (non-verbal, clear language, no jargon, closure)
    general_labels = [
        "uses simple language",
        "avoids medical jargon",
        "gives time to respond",
        "asks if parent has questions",
        "checks understanding",
        "provides closure"
    ]
    evaluation["general_communication_score"] = min(3.0, evaluate_criterion(
        doctor_messages, "communication", general_labels) / 10 * 3)

    # Calculate total
    evaluation["total_score"] = round(
        evaluation["setting_score"] +
        evaluation["perception_invitation_score"] +
        evaluation["knowledge_score"] +
        evaluation["empathy_summary_score"] +
        evaluation["general_communication_score"],
        1
    )

    # Generate feedback
    feedback_parts = []
    if evaluation["knowledge_score"] >= 2.5:
        feedback_parts.append(
            "✅ Excellent explanation of Celiac Disease and dietary management")
    elif evaluation["knowledge_score"] >= 1.5:
        feedback_parts.append(
            "⚠️ Good medical knowledge, but could provide more details about gluten-free diet")
    else:
        feedback_parts.append(
            "❌ Need to improve explanation of diagnosis and treatment")

    if evaluation["empathy_summary_score"] >= 0.8:
        feedback_parts.append("✅ Showed good empathy and emotional support")
    else:
        feedback_parts.append(
            "⚠️ Try to show more empathy and respond to parent's emotions")

    if evaluation["general_communication_score"] >= 2.5:
        feedback_parts.append(
            "✅ Clear communication with appropriate language")
    else:
        feedback_parts.append(
            "⚠️ Use simpler language and avoid medical jargon")

    evaluation["feedback"] = "\n".join(feedback_parts)

    return jsonify(evaluation)


# =====================================================
# Run the application
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting OSCE Simulation Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
