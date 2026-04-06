const chatMessages = document.getElementById("chat-messages");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const suggestButtons = document.querySelectorAll(".suggest-btn");

function addMessage(text, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "bubble";
    bubbleDiv.textContent = text;

    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "message bot";
    typingDiv.id = "typing-indicator";

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "bubble";

    const typing = document.createElement("div");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";

    bubbleDiv.appendChild(typing);
    typingDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typing = document.getElementById("typing-indicator");
    if (typing) typing.remove();
}

async function sendMessage(customMessage = null) {
    const message = customMessage || userInput.value.trim();

    if (!message) return;

    addMessage(message, "user");
    if (!customMessage) {
        userInput.value = "";
    }

    addTypingIndicator();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        removeTypingIndicator();
        addMessage(data.response || "Xin lỗi, tôi chưa thể trả lời lúc này.", "bot");
    } catch (error) {
        removeTypingIndicator();
        addMessage("Đã xảy ra lỗi khi kết nối tới trợ lý AI.", "bot");
    }
}

sendBtn.addEventListener("click", () => sendMessage());

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

suggestButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
        const text = btn.textContent.trim();
        sendMessage(text);
    });
});

window.addEventListener("DOMContentLoaded", () => {
    addMessage(
        "Xin chào, tôi là trợ lý AI của BKstore. Tôi có thể giúp bạn tìm sách, báo giá, kiểm tra tồn kho hoặc trả lời câu hỏi về mua hàng.",
        "bot"
    );
});