from typing import Optional
from openai import OpenAI

SUPPORTED = {
    "de",  # Deutsch
    "en",  # Englisch
    "fr",  # Französisch
    "tr",  # Türkisch
    "zh",  # Chinesisch (Mandarin)
    "es",  # Spanisch
    "hi",  # Hindi
    "ar",  # Arabisch
    "bn",  # Bengalisch
    "pt",  # Portugiesisch
    "ru",  # Russisch
    "ja",  # Japanisch
    "it",  # Italienisch
    "nl",  # Niederländisch
    "sv",  # Schwedisch
    "pl",  # Polnisch
    "ko",  # Koreanisch
    "fa",  # Persisch
    "cs",  # Tschechisch
    "el",  # Griechisch
    "he",  # Hebräisch
}

EDIT_CMDS = {
    "de": {
        "ändern", "korrigieren", "korrektur", "update",
        "berichtigung", "modifizieren", "anpassen", "überarbeiten", "aktualisieren"
    },
    "en": {
        "change", "edit", "update", "correct", "fix",
        "modify", "adjust", "revise", "amend"
    },
    "fr": {
        "changer", "modifier", "corriger", "mise à jour",
        "rectifier", "réviser", "ajuster"
    },
    "tr": {
        "değiştir", "düzelt", "güncelle", "düzeltme",
        "revize", "uyarlamak"
    },
    "zh": {
        "更改", "修改", "更新", "纠正", "修订", "调整"
    },
    "es": {
        "cambiar", "editar", "actualizar", "corregir",
        "modificar", "ajustar", "revisar"
    },
    "hi": {
        "बदलें", "सुधारें", "अपडेट", "संशोधित",
        "समायोजित", "संशोधन"
    },
    "ar": {
        "تغيير", "تعديل", "تحديث", "تصحيح", "مراجعة", "تسوية"
    },
    "bn": {
        "পরিবর্তন", "সংশোধন", "আপডেট", "সম্পাদনা",
        "সমন্বয়", "পরিমার্জন"
    },
    "pt": {
        "alterar", "editar", "atualizar", "corrigir",
        "modificar", "ajustar", "revisar", "emendar"
    },
    "ru": {
        "изменить", "редактировать", "обновить", "исправить",
        "корректировать", "модифицировать", "пересмотреть"
    },
    "ja": {
        "変更", "修正", "更新", "改訂", "調整"
    },
    "it": {
        "cambiare", "modificare", "correggere", "aggiornare",
        "rettificare", "revisione", "adattare"
    },
    "nl": {
        "wijzigen", "bewerken", "bijwerken", "corrigeren",
        "aanpassen", "herzien"
    },
    "sv": {
        "ändra", "redigera", "uppdatera", "korrigera",
        "revidera", "justera"
    },
    "pl": {
        "zmienić", "edytować", "zaktualizować", "poprawić",
        "modyfikować", "skorygować", "dostosować"
    },
    "ko": {
        "변경", "수정", "업데이트", "교정", "조정", "개정"
    },
    "fa": {
        "تغییر", "اصلاح", "به‌روزرسانی", "تصحیح",
        "تعدیل", "بازنگری"
    },
    "cs": {
        "změnit", "upravit", "aktualizovat", "opravit",
        "revidovat", "přizpůsobit"
    },
    "el": {
        "αλλαγή", "τροποποίηση", "ενημέρωση", "διόρθωση",
        "αναθεώρηση", "προσαρμογή"
    },
    "he": {
        "לשנות", "לעדכן", "לתקן", "עריכה",
        "התאמה", "סקירה"
    },
}

final_msgs = {
    "de": "Der Vorgang ist abgeschlossen. Vielen Dank für Ihre Übermittlung!",
    "en": "The process is complete. Thank you for your submission!",
    "fr": "Le processus est terminé. Merci pour votre soumission !",
    "tr": "İşlem tamamlandı. Gönderiniz için teşekkür ederiz!",
    "zh": "流程已完成。感谢您的提交！",
    "es": "El proceso está completo. ¡Gracias por su envío!",
    "hi": "प्रक्रिया पूरी हो गई है। आपके सबमिशन के लिए धन्यवाद!",
    "ar": "اكتملت العملية. شكرًا لتقديمك!",
    "bn": "প্রক্রিয়া সম্পূর্ণ হয়েছে। আপনার জমার জন্য ধন্যবাদ!",
    "pt": "O processo foi concluído. Obrigado pelo seu envio!",
    "ru": "Процесс завершён. Спасибо за вашу подачу!",
    "ja": "手続きが完了しました。ご提出いただきありがとうございます！",
    "it": "Il processo è completo. Grazie per la tua presentazione!",
    "nl": "Het proces is voltooid. Bedankt voor uw inzending!",
    "sv": "Processen är klar. Tack för din inlämning!",
    "pl": "Proces został zakończony. Dziękujemy za przesłanie!",
    "ko": "절차가 완료되었습니다. 제출해 주셔서 감사합니다!",
    "fa": "فرایند تکمیل شد. از ارسال شما سپاسگزاریم!",
    "cs": "Proces je dokončen. Děkujeme za vaše podání!",
    "el": "Η διαδικασία ολοκληρώθηκε. Σας ευχαριστούμε για την υποβολή σας!",
    "he": "התהליך הושלם. תודה על ההגשה שלך!"
}

download_button_msgs = {
    "de": "PDF erzeugen & herunterladen",
    "en": "Generate & Download PDF",
    "fr": "Générer et télécharger le PDF",
    "tr": "PDF oluştur ve indir",
    "zh": "生成并下载 PDF",
    "es": "Generar y descargar PDF",
    "hi": "PDF बनाएं और डाउनलोड करें",
    "ar": "إنشاء وتنزيل ملف PDF",
    "bn": "PDF তৈরি ও ডাউনলোড করুন",
    "pt": "Gerar e baixar PDF",
    "ru": "Создать и скачать PDF",
    "ja": "PDFを生成してダウンロード",
    "it": "Genera e scarica PDF",
    "nl": "PDF genereren en downloaden",
    "sv": "Skapa och ladda ner PDF",
    "pl": "Wygeneruj i pobierz PDF",
    "ko": "PDF 생성 및 다운로드",
    "fa": "تولید و دانلود PDF",
    "cs": "Vytvořit a stáhnout PDF",
    "el": "Δημιουργία και λήψη PDF",
    "he": "צור והורד PDF"
}

pdf_file_msgs = {
    "de": "Dein ausgefülltes Formular",
    "en": "Your completed form",
    "fr": "Votre formulaire rempli",
    "tr": "Doldurulmuş formunuz",
    "zh": "您填写的表格",
    "es": "Su formulario completado",
    "hi": "आपका भरा हुआ फॉर्म",
    "ar": "نموذجك المكتمل",
    "bn": "আপনার পূরণ করা ফর্ম",
    "pt": "Seu formulário preenchido",
    "ru": "Ваш заполненный бланк",
    "ja": "あなたの記入済みフォーム",
    "it": "Il tuo modulo compilato",
    "nl": "Uw ingevulde formulier",
    "sv": "Ditt ifyllda formulär",
    "pl": "Twój wypełniony formularz",
    "ko": "작성한 양식",
    "fa": "فرم تکمیل‌شده شما",
    "cs": "Váš vyplněný formulář",
    "el": "Η συμπληρωμένη φόρμα σας",
    "he": "הטופס המלא שלך"  
}

files_msgs = {
    "de": "Ihre Dateien",
    "en": "Your Files",
    "fr": "Vos fichiers",
    "tr": "Dosyalarınız",
    "zh": "您的文件",
    "es": "Sus archivos",
    "hi": "आपकी फ़ाइलें",
    "ar": "ملفاتك",
    "bn": "আপনার ফাইলসমূহ",
    "pt": "Seus arquivos",
    "ru": "Ваши файлы",
    "ja": "あなたのファイル",
    "it": "I tuoi file",
    "nl": "Uw bestanden",
    "sv": "Dina filer",
    "pl": "Twoje pliki",
    "ko": "귀하의 파일",
    "fa": "فایل‌های شما",
    "cs": "Vaše soubory",
    "el": "Τα αρχεία σας",
    "he": "הקבצים שלך"
}




def translate_from_de(text_de: str, target_lang: str, client: Optional[OpenAI] = None, model: str = "gpt-4.1-mini") -> str:
    """
    Übersetzt 'text_de' von Deutsch -> target_lang (ISO-639-1).
    - Platzhalter {so_was} / {{so_was}} / <TAGS> bleiben unverändert.
    - Juristische Begriffe wie GmbH, AG, UG, OHG, KG, e.K. usw. niemals übersetzen.
    - Bei target_lang == 'de' oder unbekanntem Code -> Rückgabe = text_de (fail-safe).
    """
    tgt = (target_lang or "de").lower()
    if tgt not in SUPPORTED or tgt == "de" or not text_de:
        return text_de

    client = client or OpenAI()

    system_prompt = (
        "You are a precise translator. Translate from German into the target language.\n"
        f"- Target language (ISO 639-1): {tgt}\n"
        "- Keep placeholders and variables exactly as-is: {like_this}, {{like_this}}, <TAGS>, $VARS, %(fmt)s.\n"
        "- Do not translate URLs, emails, codes, or content inside {{double braces}}.\n"
        "- Do not translate legal/corporate terms such as GmbH, AG, UG, OHG, KG, e.K., mbH.\n"
        "- Preserve punctuation, line breaks, and Markdown.\n"
        "- Style: concise, polite, clear."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "input_text", "text": text_de}]},
        ],
        temperature=0.0
    )

    return (resp.output_text or "").strip()

def translate_to_de(text_src: str, source_lang: str, client: Optional[OpenAI] = None, model: str = "gpt-4.1-mini") -> str:
    """
    Übersetzt 'text_src' von source_lang -> Deutsch.
    - Platzhalter {so_was} / {{so_was}} / <TAGS> bleiben unverändert.
    - Juristische Begriffe wie GmbH, AG, UG, OHG, KG, e.K. usw. niemals übersetzen.
    - Wenn source_lang == 'de' oder unbekannt -> Rückgabe = text_src (fail-safe).
    """
    src = (source_lang or "de").lower()
    if src not in SUPPORTED or src == "de" or not text_src:
        return text_src

    client = client or OpenAI()

    system_prompt = (
        "You are a precise translator. Translate into **German** from the given source language.\n"
        f"- Source language (ISO 639-1): {src}\n"
        "- Keep placeholders and variables exactly as-is: {like_this}, {{like_this}}, <TAGS>, $VARS, %(fmt)s.\n"
        "- Do not translate URLs, emails, codes, or content inside {{double braces}}.\n"
        "- Do not translate legal/corporate terms such as GmbH, AG, UG, OHG, KG, e.K., mbH.\n"
        "- Preserve punctuation, line breaks, and Markdown.\n"
        "- Style: concise, polite, clear German."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "input_text", "text": text_src}]},
        ],
        temperature=0.0
    )

    return (resp.output_text or "").strip()
