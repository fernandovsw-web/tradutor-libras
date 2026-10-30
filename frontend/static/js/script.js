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

// Largura do frame enviado ao servidor (a altura segue a proporção da câmera).
// Menor = mais frames por segundo, o que ajuda o modelo de sinais em movimento.
const FRAME_WIDTH = 320;
// Intervalo mínimo entre envios (~16 frames por segundo no máximo).
const MIN_INTERVAL_MS = 60;
// Espera após um erro de comunicação, para não sobrecarregar o servidor.
const ERROR_RETRY_MS = 1000;
 
// Identifica esta aba no servidor: o modelo de movimento precisa guardar os
// últimos frames de cada pessoa entre uma requisição e outra.
const sessionId = crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

let cameraStream = null;
let previewUrl = null;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function showResult(dados) {
    // O servidor está gravando um sinal em movimento
    if (dados.estado === "capturando") {
        prediction.textContent = "Reconhecendo movimento...";
        cameraMessage.textContent = "Movimento detectado. Conclua o sinal.";
        return;
    }
 
    // Sinal reconhecido (letra estática ou sinal em movimento): mesmo campo
    if (dados.texto) {
        const porcentagem = dados.confianca != null
            ? ` (${Math.round(dados.confianca * 100)}%)`
            : "";
        const tipo = dados.tipo === "movimento"
            ? "Sinal em movimento reconhecido"
            : "Sinal estático reconhecido";
 
        prediction.textContent = dados.texto;
        translation.textContent = dados.texto;
        cameraMessage.textContent = `${tipo}${porcentagem}.`;
        return;
    }
 
    prediction.textContent = "Nenhum sinal reconhecido";
    translation.textContent = dados.mensagem || "";
    cameraMessage.textContent = "Câmera ativa. Faça um sinal diante da câmera.";
}

// Envia UM frame e mostra a resposta. Devolve false se houve erro.
async function recognizeFrame(stream) {
    if (camera.readyState < 2 || !camera.videoWidth) {
        return true;
    }

    try {
        canvas.width = FRAME_WIDTH;
        canvas.height = Math.round(camera.videoHeight * FRAME_WIDTH / camera.videoWidth);

        context.drawImage(camera, 0, 0, canvas.width, canvas.height);

        const imagem = canvas.toDataURL("image/jpeg", 0.7);

        const resposta = await fetch("/api/traduzir", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ imagem, sessao: sessionId })
        });

        const dados = await resposta.json();
        
        // A câmera foi parada enquanto o servidor respondia: ignora o resultado
        if (cameraStream !== stream) {
            return true;
        }

        if (!resposta.ok) {
            cameraMessage.textContent =
                dados.erro || "Erro ao reconhecer o sinal.";
            return false;
        }

        showResult(dados);
        return true;
    } catch (error) {
        console.error("Erro na comunicação com o Flask:", error);
        cameraMessage.textContent =
            "Não foi possível conectar ao reconhecimento Python.";
        return false;
    }
}

// Envia frames continuamente enquanto a câmera estiver ligada. Cada envio só
// começa quando o anterior termina, então nunca há frames acumulando.
async function recognitionLoop(stream) {
    while (cameraStream === stream) {
        const inicio = performance.now();
        const ok = await recognizeFrame(stream);
 
        const espera = ok
            ? Math.max(0, MIN_INTERVAL_MS - (performance.now() - inicio))
            : ERROR_RETRY_MS;
 
        await sleep(espera);
    }
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        cameraMessage.textContent =
            "A câmera não está disponível neste endereço. Use localhost ou HTTPS.";
        return;
    }

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
            "Câmera ativa. Faça um sinal diante da câmera.";

        stopButton.disabled = false;

        // Só começa a reconhecer depois que a câmera abriu de verdade
        recognitionLoop(cameraStream);
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
        // Ao ficar null, o recognitionLoop termina sozinho
        cameraStream = null;
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
