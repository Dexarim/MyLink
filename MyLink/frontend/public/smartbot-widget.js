class SmartBotWidget extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });

    this.apiBase = this.getAttribute("api") || "http://127.0.0.1:8000";
    this.session = {};          // vacancy + resume
    this.questions = [];        // список уточняющих вопросов
    this.answers = [];          // ответы пользователя
    this.currentIndex = 0;      // текущий вопрос
    this.dialogStarted = false;

    const style = document.createElement("style");
    style.textContent = `
      .bot-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 28px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      }
      .bot-button::after {
          content: "";
          position: absolute;
          top: 10px;
          right: 10px;
          width: 12px;
          height: 12px;
          background: #10b981; /* зелёный или синий */
          border-radius: 50%;
          display: none;
          box-shadow: 0 0 6px rgba(16,185,129,0.6);
        }
        .bot-button.has-new::after {
          display: block;
        }
      .chat-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 480px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        display: none;
        flex-direction: column;
        overflow: hidden;
        font-family: "Inter", sans-serif;
      }
      .chat-header {
        background: #3b82f6;
        color: white;
        padding: 10px;
        text-align: center;
      }
      .messages {
        flex: 1;
        padding: 10px;
        overflow-y: auto;
        font-size: 14px;
        background: #fafafa;
      }
      .message.user { text-align: right; color: #1e40af; margin: 6px 0; }
      .message.bot { text-align: left; color: #111827; margin: 6px 0; }
      .chat-input {
        display: flex;
        border-top: 1px solid #ccc;
      }
      .chat-input input {
        flex: 1;
        border: none;
        padding: 10px;
        font-size: 14px;
      }
      .chat-input button {
        background: #3b82f6;
        border: none;
        color: white;
        padding: 10px 14px;
        cursor: pointer;
      }
    `;

    this.shadowRoot.innerHTML = `
      <button class="bot-button">💬</button>
      <div class="chat-container">
        <div class="chat-header">HR BOT</div>
        <div class="messages"></div>
        <div class="chat-input">
          <input type="text" placeholder="Введите сообщение..." />
          <button>➤</button>
        </div>
      </div>
    `;
    this.shadowRoot.appendChild(style);
  }

  connectedCallback() {
    const btn = this.shadowRoot.querySelector(".bot-button");
    const container = this.shadowRoot.querySelector(".chat-container");
    const messagesDiv = this.shadowRoot.querySelector(".messages");
    const input = this.shadowRoot.querySelector("input");
    const sendBtn = this.shadowRoot.querySelector(".chat-input button");

    // --- открыть/закрыть окно ---
    btn.addEventListener("click", () => {
      const opened = container.style.display === "none";
      container.style.display = container.style.display === "none" ? "flex" : "none";
      if (opened) btn.classList.remove("has-new");
    });

    // --- добавить сообщение ---
    const addMessage = (sender, text) => {
      const div = document.createElement("div");
      div.className = `message ${sender}`;
      div.textContent = text;
      messagesDiv.appendChild(div);
      messagesDiv.scrollTop = messagesDiv.scrollHeight;

      if (sender === "bot" && container.style.display === "none") {
        btn.classList.add("has-new");
      }
    };

    // --- начать диалог ---
    const startDialog = async () => {
      addMessage("bot", "Анализирую данные вакансии и резюме... ⏳");
      try {
        // шаг 1: получить gaps и базовый скор
        const res = await fetch(`${this.apiBase}/analyze/followup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.session),
        });
        const data = await res.json();
        this.questions = data.questions || [];
        this.currentIndex = 0;

        if (this.questions.length === 0) {
          addMessage("bot", "Ваше резюме отлично подходит! ✅");
        } else {
          addMessage("bot", this.questions[0].question);
        }
      } catch (err) {
        addMessage("bot", "Ошибка связи с API 😢");
      }
    };

    // --- обработать ответ пользователя ---
    const sendMessage = async () => {
      const text = input.value.trim();
      if (!text) return;
      addMessage("user", text);
      input.value = "";

      if (this.currentIndex >= this.questions.length) {
        addMessage("bot", "Все вопросы уже заданы ✅");
        return;
      }

      // сохранить ответ
      const gap = this.questions[this.currentIndex].gap;
      this.answers.push({ gap, answer: text });
      this.currentIndex++;

      if (this.currentIndex < this.questions.length) {
        // следующий вопрос
        addMessage("bot", this.questions[this.currentIndex].question);
      } else {
        // финальный расчёт
        addMessage("bot", "Обрабатываю ответы... ⚙️");
        try {
          const res = await fetch(`${this.apiBase}/analyze/final`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...this.session,
              followups: this.answers,
            }),
          });
          const data = await res.json();
          addMessage("bot", `Итоговая оценка: ${data.final_score}% 🎯`);
        } catch {
          addMessage("bot", "Ошибка при финальной оценке 😢");
        }
      }
    };

    sendBtn.addEventListener("click", sendMessage);
    input.addEventListener("keydown", (e) => e.key === "Enter" && sendMessage());

    // --- слушаем изменения атрибута data (получаем vacancy+resume) ---
    const observer = new MutationObserver(() => {
      const dataAttr = this.getAttribute("data");
      if (dataAttr && !this.dialogStarted) {
        this.dialogStarted = true;
        this.session = JSON.parse(dataAttr);
        startDialog();
      }
    });
    observer.observe(this, { attributes: true });
  }
}

customElements.define("smartbot-widget", SmartBotWidget);
