"""ينشر المشروع على Hugging Face Spaces (بيتشغّل من GitHub Actions)."""
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

SPACE_NAME = "food-calorie-ai"
INCLUDE = ["Dockerfile", "requirements-deploy.txt", ".dockerignore", "README.md",
           "src", "api", "web", "weights", "data/nutrition"]
FRONTMATTER = """---
title: Food Calorie AI
emoji: 🍽️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

"""


def fail(msg: str):
    print(f"::error::{msg}")
    sys.exit(1)


token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    print("::warning::HF_TOKEN مش موجود في Secrets الريبو، النشر اتلغى")
    sys.exit(0)

api = HfApi(token=token)
try:
    who = api.whoami()
except HfHubHTTPError as e:
    fail(f"التوكن مش شغال (اتأكد إنك نسخته كامل من غير مسافات): {e}")

user = who["name"]
role = who.get("auth", {}).get("accessToken", {}).get("role")
print(f"HF user: {user} | token type: {role}")
if role == "read":
    fail("التوكن نوعه Read. اعمل توكن جديد نوعه Write وحطّه مكان HF_TOKEN")

repo_id = f"{user}/{SPACE_NAME}"
try:
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
except HfHubHTTPError as e:
    fail(f"مش قادر أعمل الـ Space {repo_id}. لو التوكن Fine-grained، ادّيله صلاحية "
         f"'Create and write to repos' أو استخدم توكن نوعه Write: {e}")

# نجهّز فولدر فيه الملفات اللي الموقع محتاجها بس
with tempfile.TemporaryDirectory() as tmp:
    for item in INCLUDE:
        dst = os.path.join(tmp, item)
        os.makedirs(os.path.dirname(dst) or tmp, exist_ok=True)
        if os.path.isdir(item):
            shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, dst)
    readme = os.path.join(tmp, "README.md")
    with open(readme, encoding="utf-8") as f:
        body = f.read()
    with open(readme, "w", encoding="utf-8") as f:
        f.write(FRONTMATTER + body)

    try:
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          delete_patterns="*",  # نمسح أي ملفات قديمة مش موجودة دلوقتي
                          commit_message=f"Deploy {os.environ.get('GITHUB_SHA', '')[:7]}")
    except HfHubHTTPError as e:
        fail(f"الرفع على {repo_id} فشل: {e}")

host = user.lower().replace("_", "-").replace(".", "-")
url = f"https://{host}-{SPACE_NAME}.hf.space"
print(f"✅ اترفع على https://huggingface.co/spaces/{repo_id}")
print(f"🚀 الموقع (بعد 5-10 دقايق بناء): {url}")
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a", encoding="utf-8") as f:
        f.write(f"### 🚀 الموقع: {url}\n\nصفحة الـ Space (فيها حالة البناء): "
                f"https://huggingface.co/spaces/{repo_id}\n")
