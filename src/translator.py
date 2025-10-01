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

#### main.py #####

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

#### wizards.py #####
instruction_msgs = { 
    "de": "So funktioniert die Eingabe:\n\n"
          "- Antworten Sie einfach auf die Fragen in der Textbox.\n"
          "- Wenn Ihnen **Auswahlmöglichkeiten** angezeigt werden, können Sie die **Nummer** (z. B. 1 oder 1.) der Auswahl eingeben.\n"
          "- Manche Fragen sind **freiwillig** – diese können Sie mit der **Enter-Taste** überspringen.\n"
          "- Bei bestimmten Fragen können Sie zusätzlich **Dokumente hochladen**. Klicken Sie dafür einfach auf den erscheinenden Button.\n"
          "- Zu vielen Fragen werden Ihnen **weiterführende Informationen** angezeigt, welche Sie durch Klicken ausklappen können.\n"
          "- Wenn Sie eine bereits gemachte Angabe ändern möchten, schreiben Sie das einfach in die Textbox, z. B.: \"Ich möchte den angegebenen Firmennamen **ändern**.\"\n\n",

    "en": "How to enter your information:\n\n"
          "- Simply answer the questions in the textbox.\n"
          "- If you are shown **options**, you can enter the **number** (e.g., 1 or 1.) of the choice.\n"
          "- Some questions are **optional** – you can skip them by pressing **Enter**.\n"
          "- For some questions, you can also **upload documents**. Just click the button that appears.\n"
          "- Many questions come with **additional information**, which you can expand by clicking.\n"
          "- If you want to change an answer you already gave, simply write it in the textbox, e.g.: \"I would like to **change** the company name provided.\"\n\n",

    "fr": "Comment saisir vos informations :\n\n"
          "- Répondez simplement aux questions dans la zone de texte.\n"
          "- Si des **options** vous sont proposées, vous pouvez entrer le **numéro** (par ex. 1 ou 1.) du choix.\n"
          "- Certaines questions sont **facultatives** – vous pouvez les ignorer en appuyant sur **Entrée**.\n"
          "- Pour certaines questions, vous pouvez également **téléverser des documents**. Cliquez simplement sur le bouton qui apparaît.\n"
          "- De nombreuses questions sont accompagnées d’**informations complémentaires** que vous pouvez afficher en cliquant.\n"
          "- Si vous souhaitez modifier une réponse déjà donnée, écrivez-le simplement dans la zone de texte, par ex. : « Je souhaite **modifier** le nom de l’entreprise indiqué. »\n\n",

    "tr": "Bilgilerinizi nasıl gireceksiniz:\n\n"
          "- Sorulara metin kutusunda yanıt verin.\n"
          "- Eğer size **seçenekler** sunulursa, seçeneğin **numarasını** (örn. 1 veya 1.) girebilirsiniz.\n"
          "- Bazı sorular **isteğe bağlıdır** – bunları **Enter** tuşuna basarak atlayabilirsiniz.\n"
          "- Bazı sorularda ayrıca **belge yükleyebilirsiniz**. Görünen butona tıklayın.\n"
          "- Birçok soruda **ek bilgiler** gösterilir, bunları tıklayarak açabilirsiniz.\n"
          "- Daha önce verdiğiniz bir cevabı değiştirmek isterseniz, bunu metin kutusuna yazın, örn.: \"Verilen şirket adını **değiştirmek** istiyorum.\" \n\n",

    "zh": "输入说明：\n\n"
          "- 请在文本框中回答问题。\n"
          "- 如果出现**选项**，您可以输入所选项的**编号**（例如 1 或 1.）。\n"
          "- 有些问题是**可选的** – 您可以按 **Enter** 跳过。\n"
          "- 对于某些问题，您还可以**上传文件**。只需点击出现的按钮。\n"
          "- 许多问题提供**附加信息**，您可以点击展开查看。\n"
          "- 如果您想更改已填写的信息，只需在文本框中输入，例如：“我想**更改**填写的公司名称。”\n\n",

    "es": "Cómo ingresar la información:\n\n"
          "- Simplemente responda a las preguntas en el cuadro de texto.\n"
          "- Si se muestran **opciones**, puede introducir el **número** (p. ej., 1 o 1.) de la opción.\n"
          "- Algunas preguntas son **opcionales** – puede omitirlas presionando **Enter**.\n"
          "- En algunas preguntas también puede **subir documentos**. Haga clic en el botón que aparece.\n"
          "- Muchas preguntas incluyen **información adicional**, que puede desplegar haciendo clic.\n"
          "- Si desea cambiar una respuesta ya dada, escríbalo en el cuadro de texto, p. ej.: «Quiero **cambiar** el nombre de la empresa indicado.»\n\n",

    "hi": "जानकारी भरने का तरीका:\n\n"
          "- बस टेक्स्टबॉक्स में प्रश्नों का उत्तर दें।\n"
          "- यदि आपको **विकल्प** दिखाए जाते हैं, तो आप उस विकल्प की **संख्या** (जैसे 1 या 1.) दर्ज कर सकते हैं।\n"
          "- कुछ प्रश्न **वैकल्पिक** होते हैं – इन्हें आप **Enter** दबाकर छोड़ सकते हैं।\n"
          "- कुछ प्रश्नों में आप **दस्तावेज़ भी अपलोड कर सकते हैं**। बस दिखाई देने वाले बटन पर क्लिक करें।\n"
          "- कई प्रश्नों में **अतिरिक्त जानकारी** दी जाती है, जिसे आप क्लिक करके खोल सकते हैं।\n"
          "- यदि आप पहले से दी गई जानकारी बदलना चाहते हैं, तो इसे टेक्स्टबॉक्स में लिखें, जैसे: \"मैं दिए गए कंपनी नाम को **बदलना** चाहता हूँ।\"\n\n",

    "ar": "كيفية إدخال المعلومات:\n\n"
          "- أجب ببساطة عن الأسئلة في مربع النص.\n"
          "- إذا ظهرت لك **خيارات**، يمكنك إدخال **رقم** (مثل 1 أو 1.) الخيار.\n"
          "- بعض الأسئلة **اختيارية** – يمكنك تخطيها بالضغط على **Enter**.\n"
          "- لبعض الأسئلة يمكنك أيضًا **تحميل مستندات**. فقط انقر على الزر الذي يظهر.\n"
          "- العديد من الأسئلة تتضمن **معلومات إضافية** يمكنك عرضها بالنقر.\n"
          "- إذا أردت تعديل إجابة سابقة، فاكتب ذلك في مربع النص، مثل: \"أود **تغيير** اسم الشركة المذكور.\" \n\n",

    "bn": "তথ্য প্রবেশ করার নিয়ম:\n\n"
          "- টেক্সটবক্সে প্রশ্নগুলির উত্তর দিন।\n"
          "- যদি আপনাকে **বিকল্পগুলি** দেখানো হয়, আপনি নির্বাচিত বিকল্পের **সংখ্যা** (যেমন 1 বা 1.) লিখতে পারেন।\n"
          "- কিছু প্রশ্ন **ঐচ্ছিক** – আপনি **Enter** চাপ দিয়ে এগুলি এড়িয়ে যেতে পারেন।\n"
          "- কিছু প্রশ্নে আপনি **ডকুমেন্ট আপলোডও করতে পারেন**। প্রদর্শিত বোতামে ক্লিক করুন।\n"
          "- অনেক প্রশ্নে **অতিরিক্ত তথ্য** দেওয়া হয় যা ক্লিক করে দেখা যায়।\n"
          "- যদি আপনি পূর্বে দেওয়া কোনো তথ্য পরিবর্তন করতে চান, তাহলে টেক্সটবক্সে লিখুন, যেমন: \"আমি প্রদত্ত কোম্পানির নামটি **পরিবর্তন** করতে চাই।\" \n\n",

    "pt": "Como inserir as informações:\n\n"
          "- Basta responder às perguntas na caixa de texto.\n"
          "- Se forem exibidas **opções**, você pode digitar o **número** (ex.: 1 ou 1.) da opção.\n"
          "- Algumas perguntas são **opcionais** – você pode pulá-las pressionando **Enter**.\n"
          "- Em algumas perguntas, você também pode **enviar documentos**. Basta clicar no botão exibido.\n"
          "- Muitas perguntas trazem **informações adicionais**, que você pode expandir clicando.\n"
          "- Se quiser alterar uma resposta já fornecida, basta escrever na caixa de texto, ex.: \"Quero **alterar** o nome da empresa informado.\" \n\n",

    "ru": "Как ввести данные:\n\n"
          "- Просто отвечайте на вопросы в текстовом поле.\n"
          "- Если вам показаны **варианты**, вы можете ввести **номер** (например, 1 или 1.) выбранного варианта.\n"
          "- Некоторые вопросы **необязательные** – их можно пропустить, нажав **Enter**.\n"
          "- В некоторых случаях вы также можете **загрузить документы**. Просто нажмите появившуюся кнопку.\n"
          "- Многие вопросы содержат **дополнительную информацию**, которую можно раскрыть по клику.\n"
          "- Если вы хотите изменить уже введённые данные, просто напишите это в текстовом поле, например: «Я хочу **изменить** указанное название компании.»\n\n",

    "ja": "入力方法：\n\n"
          "- テキストボックスに質問の答えを入力してください。\n"
          "- **選択肢**が表示された場合は、その**番号**（例: 1 または 1.）を入力できます。\n"
          "- 一部の質問は**任意**です – **Enter**キーでスキップできます。\n"
          "- 質問によっては**書類をアップロード**することもできます。表示されたボタンをクリックしてください。\n"
          "- 多くの質問には**追加情報**があり、クリックで展開できます。\n"
          "- すでに入力した内容を変更したい場合は、テキストボックスに入力してください。例: 「入力した会社名を**変更**したい。」\n\n",

    "it": "Come inserire le informazioni:\n\n"
          "- Rispondi semplicemente alle domande nella casella di testo.\n"
          "- Se vengono mostrate delle **opzioni**, puoi digitare il **numero** (es. 1 o 1.) della scelta.\n"
          "- Alcune domande sono **facoltative** – puoi saltarle premendo **Invio**.\n"
          "- In alcune domande puoi anche **caricare documenti**. Basta cliccare sul pulsante che appare.\n"
          "- Molte domande includono **ulteriori informazioni**, che puoi espandere cliccando.\n"
          "- Se vuoi modificare una risposta già fornita, scrivilo nella casella di testo, es.: \"Vorrei **modificare** il nome dell'azienda indicato.\" \n\n",

    "nl": "Hoe informatie invoeren:\n\n"
          "- Beantwoord gewoon de vragen in het tekstvak.\n"
          "- Als er **opties** worden getoond, kunt u het **nummer** (bijv. 1 of 1.) van de keuze invoeren.\n"
          "- Sommige vragen zijn **optioneel** – u kunt deze overslaan door op **Enter** te drukken.\n"
          "- Bij sommige vragen kunt u ook **documenten uploaden**. Klik eenvoudig op de knop die verschijnt.\n"
          "- Veel vragen bevatten **aanvullende informatie**, die u kunt uitklappen door te klikken.\n"
          "- Wilt u een eerder gegeven antwoord wijzigen, typ dat gewoon in het tekstvak, bijv.: \"Ik wil de opgegeven bedrijfsnaam **wijzigen**.\" \n\n",

    "sv": "Så här fyller du i uppgifterna:\n\n"
          "- Svara på frågorna i textrutan.\n"
          "- Om du visas **alternativ**, kan du skriva in **numret** (t.ex. 1 eller 1.) för valet.\n"
          "- Vissa frågor är **frivilliga** – du kan hoppa över dem genom att trycka på **Enter**.\n"
          "- Vid vissa frågor kan du även **ladda upp dokument**. Klicka bara på knappen som visas.\n"
          "- Många frågor har **ytterligare information** som du kan expandera genom att klicka.\n"
          "- Om du vill ändra ett redan givet svar, skriv det i textrutan, t.ex.: \"Jag vill **ändra** det angivna företagsnamnet.\" \n\n",

    "pl": "Jak wprowadzić informacje:\n\n"
          "- Odpowiedz po prostu na pytania w polu tekstowym.\n"
          "- Jeśli wyświetlą się **opcje**, możesz wpisać **numer** (np. 1 lub 1.) wyboru.\n"
          "- Niektóre pytania są **opcjonalne** – możesz je pominąć naciskając **Enter**.\n"
          "- W niektórych pytaniach możesz również **przesłać dokumenty**. Kliknij wyświetlony przycisk.\n"
          "- Wiele pytań zawiera **dodatkowe informacje**, które możesz rozwinąć klikając.\n"
          "- Jeśli chcesz zmienić wcześniej podaną odpowiedź, wpisz to w polu tekstowym, np.: \"Chcę **zmienić** podaną nazwę firmy.\" \n\n",

    "ko": "입력 방법:\n\n"
          "- 텍스트 상자에 질문에 대한 답을 입력하세요.\n"
          "- **선택지**가 표시되면 선택의 **번호**(예: 1 또는 1.)를 입력할 수 있습니다.\n"
          "- 일부 질문은 **선택 사항**이며, **Enter** 키를 눌러 건너뛸 수 있습니다.\n"
          "- 일부 질문에서는 **문서를 업로드**할 수도 있습니다. 나타나는 버튼을 클릭하세요.\n"
          "- 많은 질문에는 **추가 정보**가 있으며, 클릭하여 펼칠 수 있습니다.\n"
          "- 이미 입력한 내용을 변경하려면 텍스트 상자에 입력하세요. 예: \"제공한 회사 이름을 **변경**하고 싶습니다.\" \n\n",

    "fa": "نحوه وارد کردن اطلاعات:\n\n"
          "- به سادگی به سؤالات در جعبه متن پاسخ دهید.\n"
          "- اگر **گزینه‌هایی** به شما نمایش داده شد، می‌توانید **شماره** (مثلاً 1 یا 1.) انتخاب را وارد کنید.\n"
          "- برخی سؤالات **اختیاری** هستند – می‌توانید با زدن **Enter** از آنها عبور کنید.\n"
          "- برای برخی سؤالات می‌توانید همچنین **اسناد بارگذاری کنید**. کافیست روی دکمه ظاهر شده کلیک کنید.\n"
          "- بسیاری از سؤالات دارای **اطلاعات اضافی** هستند که می‌توانید با کلیک مشاهده کنید.\n"
          "- اگر می‌خواهید پاسخی که قبلاً داده‌اید را تغییر دهید، کافیست آن را در جعبه متن بنویسید، مثلاً: «می‌خواهم نام شرکت وارد شده را **تغییر** دهم.» \n\n",

    "cs": "Jak zadávat údaje:\n\n"
          "- Jednoduše odpovězte na otázky v textovém poli.\n"
          "- Pokud se zobrazí **možnosti**, můžete zadat **číslo** (např. 1 nebo 1.) volby.\n"
          "- Některé otázky jsou **nepovinné** – můžete je přeskočit stisknutím klávesy **Enter**.\n"
          "- U některých otázek můžete také **nahrát dokumenty**. Klikněte na zobrazené tlačítko.\n"
          "- Mnoho otázek obsahuje **další informace**, které můžete rozbalit kliknutím.\n"
          "- Pokud chcete změnit již zadanou odpověď, napište to do textového pole, např.: \"Chci **změnit** uvedený název společnosti.\" \n\n",

    "el": "Πώς να εισάγετε τις πληροφορίες:\n\n"
          "- Απαντήστε στις ερωτήσεις στο πλαίσιο κειμένου.\n"
          "- Αν σας εμφανιστούν **επιλογές**, μπορείτε να πληκτρολογήσετε τον **αριθμό** (π.χ. 1 ή 1.) της επιλογής.\n"
          "- Ορισμένες ερωτήσεις είναι **προαιρετικές** – μπορείτε να τις παραλείψετε πατώντας το **Enter**.\n"
          "- Σε ορισμένες ερωτήσεις μπορείτε επίσης να **ανεβάσετε έγγραφα**. Απλά κάντε κλικ στο κουμπί που εμφανίζεται.\n"
          "- Πολλές ερωτήσεις περιλαμβάνουν **επιπλέον πληροφορίες** που μπορείτε να εμφανίσετε κάνοντας κλικ.\n"
          "- Αν θέλετε να αλλάξετε μια απάντηση που έχετε ήδη δώσει, απλώς γράψτε το στο πλαίσιο κειμένου, π.χ.: «Θέλω να **αλλάξω** το δηλωμένο όνομα της εταιρείας.» \n\n",

    "he": "כיצד להזין את המידע:\n\n"
          "- פשוט השב על השאלות בתיבת הטקסט.\n"
          "- אם מוצגות לך **אפשרויות**, תוכל להזין את **המספר** (לדוגמה 1 או 1.) של האפשרות.\n"
          "- חלק מהשאלות הן **רשות** – ניתן לדלג עליהן באמצעות לחיצה על **Enter**.\n"
          "- בחלק מהשאלות ניתן גם **להעלות מסמכים**. פשוט לחץ על הכפתור שמופיע.\n"
          "- שאלות רבות כוללות **מידע נוסף** שניתן להרחיב על ידי לחיצה.\n"
          "- אם ברצונך לשנות תשובה שכבר נתת, פשוט כתוב זאת בתיבת הטקסט, לדוגמה: \"אני רוצה **לשנות** את שם החברה שסיפקתי.\" \n\n"
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
