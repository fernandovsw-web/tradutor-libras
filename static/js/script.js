
const camera = document.getElementById("camera");
const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const cameraStatus = document.getElementById("camera-status");
const cameraMessage = document.getElementById("camera-message");
const cameraPlaceholder = document.getElementById("camera-placeholder");

const signImage = document.getElementById("sign-image");
const imagePreview = document.getElementById("image-preview");
const imagePreviewBox = document.getElementById("image-preview-box");
const removeImageButton = document.getElementById("remove-image");

const prediction = document.getElementById("prediction");
const translation = document.getElementById("translation");

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

let recognitionTimer = null;
let sendingFrame = false;

let cameraStream = null;
let previewUrl = null;

async function recognizeFrame() {
    if (!cameraStream || camera.readyState < 2 || sendingFrame) {
        return;
    }

    sendingFrame = true;

    try {
        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;

        context.drawImage(camera, 0, 0, canvas.width, canvas.height);

        const imagem = canvas.toDataURL("image/jpeg", 0.65);

        const resposta = await fetch("/api/reconhecer", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ imagem })
        });

        const dados = await resposta.json();

        if (!resposta.ok) {
            cameraMessage.textContent =
                dados.erro || "Erro ao reconhecer o sinal.";
            return;
        }

        if (dados.letra) {
            prediction.textContent = dados.letra;
            translation.textContent = dados.letra;
        } else {
            prediction.textContent = "Nenhuma letra reconhecida";
            translation.textContent = dados.mensagem;
        }
    } catch (error) {
        console.error("Erro na comunicação com o Flask:", error);
        cameraMessage.textContent =
            "Não foi possível conectar ao reconhecimento Python.";
    } finally {
        sendingFrame = false;
    }
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        cameraMessage.textContent =
            "A câmera não está disponível neste endereço. Use localhost ou HTTPS.";
        return;
    }
recognitionTimer = setInterval(recognizeFrame, 500);
    startButton.disabled = true;
    cameraMessage.textContent = "Solicitando acesso à câmera...";

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        camera.srcObject = cameraStream;
        await camera.play();

        camera.classList.add("visible");
        cameraPlaceholder.classList.add("hidden");

        cameraStatus.textContent = "Câmera ativa";
        cameraStatus.classList.add("active");
        cameraMessage.textContent =
            "Câmera ativa. A previsão depende da integração com o modelo Python.";

        stopButton.disabled = false;
    } catch (error) {
        console.error("Erro ao abrir a câmera:", error);

        if (error.name === "NotAllowedError") {
            cameraMessage.textContent =
                "Acesso negado. Permita o uso da câmera nas configurações do navegador.";
        } else if (error.name === "NotFoundError") {
            cameraMessage.textContent = "Nenhuma câmera foi encontrada.";
        } else {
            cameraMessage.textContent =
                "Não foi possível abrir a câmera. Verifique se ela está sendo usada por outro programa.";
        }

        cameraStatus.textContent = "Câmera desligada";
        cameraStatus.classList.remove("active");
        startButton.disabled = false;
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
if (recognitionTimer) {
    clearInterval(recognitionTimer);
    recognitionTimer = null;
}
    camera.srcObject = null;
    camera.classList.remove("visible");
    cameraPlaceholder.classList.remove("hidden");

    cameraStatus.textContent = "Câmera desligada";
    cameraStatus.classList.remove("active");
    cameraMessage.textContent = "Câmera desligada.";

    startButton.disabled = false;
    stopButton.disabled = true;
}

function clearImage() {
    signImage.value = "";
    imagePreview.removeAttribute("src");
    imagePreviewBox.hidden = true;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

startButton?.addEventListener("click", startCamera);
stopButton?.addEventListener("click", stopCamera);
removeImageButton?.addEventListener("click", clearImage);

signImage?.addEventListener("change", () => {
    const file = signImage.files?.[0];

    if (!file) return;

    const allowedTypes = ["image/png", "image/jpeg", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
        cameraMessage.textContent =
            "Escolha uma imagem PNG, JPG ou WEBP.";
        clearImage();
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        cameraMessage.textContent =
            "A imagem deve ter no máximo 5 MB.";
        clearImage();
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    imagePreviewBox.hidden = false;
});

window.addEventListener("pagehide", stopCamera);
