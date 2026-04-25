from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import os

app = Flask(__name__)
CORS(app)

# Claude API Client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Global session
conversation_history = []
case_data = {}

# =====================================================
# System Prompt
# =====================================================


def get_system_prompt():
    return """أنتِ أم اسمك حسناء، لديكِ طفل عمره 3 سنوات اسمه علي.
علي يعاني من إسهال مزمن وألم في البطن منذ أكثر من سنة.
اليوم جئتِ للعيادة لمعرفة نتائج المنظار والخزعة.

شخصيتك:
- أم قلقة ومهتمة جداً بصحة ولدها
- تتكلمين بجمل قصيرة وبسيطة (جملة أو اثنتين فقط)
- تستخدمين لغة يومية عادية بدون مصطلحات طبية
- تبدين القلق والخوف على ولدها
- ما أحد من العائلة عنده نفس المرض
- والد علي مسافر وأنتِ لوحدك في العيادة

القواعد:
- جاوبين بجملة أو جملتين فقط، لا أكثر
- ردودك طبيعية وعفوية مثل أم حقيقية
- إذا أخبرك الدكتور بالتشخيص أبدي القلق واسألي عن التفاصيل
- اسألي عن: السبب، العلاج، الأكل المسموح والممنوع، هل المرض دائم؟
- لا تعطين معلومات طبية، فقط اسألي وأبدي ردود فعل طبيعية"""

# =====================================================
# Routes
# =====================================================


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "claude-sonnet-4-20250514"})


@app.route("/update-case", methods=["POST"])
def update_case():
    global conversation_history, case_data

    case_data = request.json or {}

    initial_message = "السلام عليكم دكتور.. أنا أم علي. جيت اليوم عشان أعرف نتائج الفحوصات. هل طلع فيه شي؟"

    conversation_history = [
        {"role": "user", "content": initial_message}
    ]

    return jsonify({
        "status": "success",
        "initial_message": initial_message
    })


@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    data = request.json
    doctor_message = data.get("message", "").strip()

    if not doctor_message:
        return jsonify({"error": "رسالة فارغة"}), 400

    # إذا ما في محادثة، ابدأ تلقائياً
    if not conversation_history:
        initial_message = "السلام عليكم دكتور.. أنا أم علي. جيت اليوم عشان أعرف نتائج الفحوصات. هل طلع فيه شي؟"
        conversation_history.append(
            {"role": "user", "content": initial_message})

    conversation_history.append(
        {"role": "assistant", "content": doctor_message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            system=get_system_prompt(),
            messages=conversation_history
        )
        mother_reply = response.content[0].text.strip()
    except Exception as e:
        print(f"Claude API error: {e}")
        mother_reply = "آسفة دكتور.. ما فهمت كلامك. ممكن توضح لي أكثر؟"

    conversation_history.append({"role": "user", "content": mother_reply})

    return jsonify({
        "reply": mother_reply,
        "conversation_length": len(conversation_history)
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    # Collect all doctor messages
    doctor_messages = []
    for i, msg in enumerate(conversation_history):
        if msg["role"] == "assistant":
            doctor_messages.append(msg["content"])

    if not doctor_messages:
        return jsonify({"error": "لا يوجد محادثة للتقييم"}), 400

    full_conversation = "\n".join([
        f"{'الطبيب' if m['role'] == 'assistant' else 'الأم'}: {m['content']}"
        for m in conversation_history
    ])

    eval_prompt = f"""أنت ممتحن OSCE متخصص. قيّم أداء الطالب في هذه المحادثة بناءً على mark sheet التالي:

المحادثة:
{full_conversation}

قيّم كل محور وأعطِ درجة دقيقة:

S: Setting (max 1 درجة)
- تعريف النفس، ضمان الخصوصية، بناء العلاقة، سؤال إذا تحتاج أحد معها

P+I: Perception & Invitation (max 2 درجة)
- سؤال الأم عما تعرفه، والسؤال إذا تريد معرفة التشخيص

K: Knowledge (max 3 درجات) - 0.75 لكل نقطة:
- شرح تشخيص الداء البطني بناءً على الأعراض والخزعة
- شرح أن السبب هو تفاعل مناعي مع الغلوتين
- شرح أهمية الحمية الخالية من الغلوتين كعلاج أساسي
- توضيح أن المرض مزمن مدى الحياة

E+S: Empathy & Summary (max 1 درجة)
- التعاطف مع مشاعر الأم، التلخيص، الإغلاق، ترتيب متابعة

General Communication (max 3 درجات) - 0.5 لكل نقطة:
- التواصل غير اللفظي
- إعطاء الأم وقتاً للتعبير
- السؤال إذا عندها أسئلة
- التحقق من الفهم
- استخدام لغة بسيطة بدون مصطلحات
- الإغلاق (شكراً)

أجب بهذا الشكل JSON فقط بدون أي نص إضافي:
{{
  "setting_score": X,
  "perception_invitation_score": X,
  "knowledge_score": X,
  "empathy_summary_score": X,
  "general_communication_score": X,
  "total_score": X,
  "feedback_ar": "ملاحظات تفصيلية بالعربي",
  "strengths": ["نقطة قوة 1", "نقطة قوة 2"],
  "improvements": ["نقطة تحسين 1", "نقطة تحسين 2"]
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        import json
        result_text = response.content[0].text.strip()
        # Remove markdown if present
        result_text = result_text.replace(
            "```json", "").replace("```", "").strip()
        evaluation = json.loads(result_text)
    except Exception as e:
        print(f"Evaluation error: {e}")
        evaluation = {
            "setting_score": 0,
            "perception_invitation_score": 0,
            "knowledge_score": 0,
            "empathy_summary_score": 0,
            "general_communication_score": 0,
            "total_score": 0,
            "feedback_ar": "حدث خطأ في التقييم",
            "strengths": [],
            "improvements": []
        }

    return jsonify(evaluation)


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
