/*
script.js
 
PARA QUE SERVE
--------------
É o JavaScript da página tradutor.html. Liga a câmera do usuário, envia os
quadros (frames) para o servidor Flask e mostra na tela o sinal que o
backend reconheceu. Também cuida da prévia da imagem de referência, que é só
para o usuário praticar e não é enviada ao servidor.
 
O reconhecimento em si NÃO acontece aqui: este arquivo só captura imagens e
exibe respostas. Quem decide o que é cada sinal é o traduzir.py.
 
COMO FUNCIONA (passo a passo)
-----------------------------
1. Ao clicar em "Iniciar câmera" (startCamera), o navegador pede permissão,
   o vídeo aparece na tela e o laço de reconhecimento começa.
2. O laço (recognitionLoop) repete sem parar enquanto a câmera estiver
   ligada: desenha o frame atual em um canvas pequeno, converte para JPEG em
   base64 e envia por POST para /api/traduzir (recognizeFrame).
3. O servidor responde com o resultado e showResult o escreve nos campos da
   página. Só depois dessa resposta o próximo frame é enviado, então os
   envios nunca se acumulam.
4. Ao clicar em "Parar câmera" (stopCamera), a câmera é desligada e o laço
   termina sozinho.
 
COMO FICA A COMUNICAÇÃO COM O traduzir.py
-----------------------------------------
Envio (JSON, POST /api/traduzir):
    imagem  frame em "data:image/jpeg;base64,..."
    sessao  id desta aba (veja sessionId abaixo)
 
Resposta (JSON). Os campos que este arquivo usa são:
    estado     "livre" ou "capturando". Com "capturando", o servidor está
               gravando um sinal em movimento e ainda não tem resultado.
    tipo       "estatico" (letra com a mão parada), "movimento" (sinal com
               movimento) ou "nenhum" (nada reconhecido).
    texto      a letra ou o sinal reconhecido. É null quando não há
               reconhecimento.
    confianca  de 0 a 1; vira porcentagem na mensagem. Pode ser null.
    mensagem   frase explicativa do servidor (ex.: "Nenhuma mão detectada.").
 
Se o servidor recusar a requisição (HTTP 400), a resposta traz apenas o campo
"erro", que é mostrado ao usuário. O campo "letra" também vem na resposta,
mas só por compatibilidade com a rota antiga, e este arquivo não o usa.
 
Se o traduzir.py mudar algum desses nomes ou valores, showResult precisa
mudar junto.
 
COMO USAR
---------
tradutor.html carrega este arquivo com "defer":
 
    <script src="{{ url_for('static', filename='js/script.js') }}" defer></script>
 
Por isso o arquivo precisa ficar em static/js/script.js. A rota /api/traduzir
precisa estar registrada no Flask (traduzir_bp, em traduzir.py).
 
DEPENDÊNCIAS
------------
- Os ids do tradutor.html (listados na seção ELEMENTOS DA PÁGINA).
- Um navegador com acesso à câmera. Isso exige abrir o site por localhost ou
  por HTTPS; em um endereço http comum a câmera fica indisponível.
- Nenhuma biblioteca externa.
*/
 
// ---------------------------------------------------------------------------
// ELEMENTOS DA PÁGINA (ids definidos em tradutor.html)
// ---------------------------------------------------------------------------
 
// Painel da câmera
const camera = document.getElementById("camera");
const startButton = document.getElementById("start-camera");
const stopButton = document.getElementById("stop-camera");
const cameraStatus = document.getElementById("camera-status");
const cameraMessage = document.getElementById("camera-message");
const cameraPlaceholder = document.getElementById("camera-placeholder");

// Painel da imagem de referência
const signImage = document.getElementById("sign-image");
const imagePreview = document.getElementById("image-preview");
const imagePreviewBox = document.getElementById("image-preview-box");
const removeImageButton = document.getElementById("remove-image");

// Caixas de resultado
const prediction = document.getElementById("prediction");
const translation = document.getElementById("translation");

// Canvas fora da página, usado só para reduzir cada frame antes do envio
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

// ---------------------------------------------------------------------------
// CONFIGURAÇÕES (ajuste aqui se precisar calibrar)
// ---------------------------------------------------------------------------

// Largura do frame enviado ao servidor (a altura segue a proporção da câmera).
// Menor = mais frames por segundo, o que ajuda o modelo de sinais em movimento.
const FRAME_WIDTH = 320;
// Intervalo mínimo entre envios (~16 frames por segundo no máximo).
const MIN_INTERVAL_MS = 60;
// Espera após um erro de comunicação, para não sobrecarregar o servidor.
const ERROR_RETRY_MS = 1000;

// ---------------------------------------------------------------------------
// SESSÃO E ESTADO
// ---------------------------------------------------------------------------

// Identifica esta aba no servidor: o modelo de movimento precisa guardar os
// últimos frames de cada pessoa entre uma requisição e outra. É enviado em
// todo frame como "sessao"; o traduzir.py usa o id para achar o histórico
// desta aba (classe Sessao) e descarta sessões paradas por 60 segundos.
// O id é gerado ao abrir a página, então recarregar a página começa uma
// sessão nova. Se o navegador não tiver crypto.randomUUID, monta um id
// parecido com data e número aleatório.
const sessionId = crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

// Fluxo de vídeo da câmera; null com a câmera desligada. O laço de
// reconhecimento usa este valor para saber quando parar.
let cameraStream = null;
// Endereço temporário da prévia da imagem de referência (precisa ser liberado)
let previewUrl = null;

// Pausa assíncrona: "await sleep(100)" espera 100 ms
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// RESULTADO: escreve a resposta do servidor na tela
// ---------------------------------------------------------------------------
 
// Recebe o JSON do traduzir.py (campos estado, tipo, texto, confianca e
// mensagem; veja o cabeçalho) e decide o que mostrar.
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
        // "tipo" só serve para dizer ao usuário que espécie de sinal foi reconhecido
        const tipo = dados.tipo === "movimento"
            ? "Sinal em movimento reconhecido"
            : "Sinal estático reconhecido";
 
        prediction.textContent = dados.texto;
        translation.textContent = dados.texto;
        cameraMessage.textContent = `${tipo}${porcentagem}.`;
        return;
    }
    
    // Sem texto: o servidor não reconheceu nada. A frase dele (ex.: "Nenhuma
    // mão detectada.") vai para a caixa de tradução.
    prediction.textContent = "Nenhum sinal reconhecido";
    translation.textContent = dados.mensagem || "";
    cameraMessage.textContent = "Câmera ativa. Faça um sinal diante da câmera.";
}

// ---------------------------------------------------------------------------
// ENVIO DE UM FRAME
// ---------------------------------------------------------------------------

// Envia UM frame e mostra a resposta. Devolve false se houve erro.
// O parâmetro "stream" é a câmera que estava ligada quando o envio começou.
async function recognizeFrame(stream) {
    // Vídeo ainda não tem imagem pronta: pula este frame sem contar como erro
    if (camera.readyState < 2 || !camera.videoWidth) {
        return true;
    }

    try {
        // Reduz o frame para FRAME_WIDTH de largura, mantendo a proporção.
        // A imagem NÃO é espelhada aqui: o servidor espelha (cv2.flip) para
        // ficar igual à coleta e ao treino.
        canvas.width = FRAME_WIDTH;
        canvas.height = Math.round(camera.videoHeight * FRAME_WIDTH / camera.videoWidth);

        context.drawImage(camera, 0, 0, canvas.width, canvas.height);

        // JPEG com qualidade 0.7, em texto base64 ("data:image/jpeg;base64,...")
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

        // Erro do servidor (ex.: HTTP 400): a resposta traz o campo "erro"
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

// ---------------------------------------------------------------------------
// LAÇO DE RECONHECIMENTO
// ---------------------------------------------------------------------------

// Envia frames continuamente enquanto a câmera estiver ligada. Cada envio só
// começa quando o anterior termina, então nunca há frames acumulando.
//
// A cada volta:
// - ok: espera só o que faltar para completar MIN_INTERVAL_MS desde o início
//   do envio. Se o servidor demorou mais que isso, não espera nada.
// - erro: espera ERROR_RETRY_MS inteiro antes de tentar de novo.
//
// O laço termina quando cameraStream deixa de ser o fluxo que o iniciou, ou
// seja, depois de stopCamera. Não existe outro botão de parar.
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

// ---------------------------------------------------------------------------
// CÂMERA: ligar e desligar
// ---------------------------------------------------------------------------

async function startCamera() {
    // getUserMedia só existe em localhost ou HTTPS
    if (!navigator.mediaDevices?.getUserMedia) {
        cameraMessage.textContent =
            "A câmera não está disponível neste endereço. Use localhost ou HTTPS.";
        return;
    }

    startButton.disabled = true;
    cameraMessage.textContent = "Solicitando acesso à câmera...";

    try {
        // Pede a câmera frontal, sem áudio
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        camera.srcObject = cameraStream;
        await camera.play();

        // Mostra o vídeo e esconde o aviso "Sua câmera aparecerá aqui"
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

    // Volta a tela ao estado inicial: esconde o vídeo e mostra o aviso
    camera.srcObject = null;
    camera.classList.remove("visible");
    cameraPlaceholder.classList.remove("hidden");

    cameraStatus.textContent = "Câmera desligada";
    cameraStatus.classList.remove("active");
    cameraMessage.textContent = "Câmera desligada.";

    startButton.disabled = false;
    stopButton.disabled = true;
}

// ---------------------------------------------------------------------------
// IMAGEM DE REFERÊNCIA (só prévia, nada é enviado ao servidor)
// ---------------------------------------------------------------------------

// Esconde a prévia, limpa o campo de arquivo e libera o endereço temporário
function clearImage() {
    signImage.value = "";
    imagePreview.removeAttribute("src");
    imagePreviewBox.hidden = true;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }
}

// ---------------------------------------------------------------------------
// EVENTOS
// ---------------------------------------------------------------------------
 
// O "?." evita erro se algum elemento não existir na página
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

    // Libera a prévia anterior antes de criar a nova
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    imagePreviewBox.hidden = false;
});

// Desliga a câmera ao sair ou fechar a página, para a luz da câmera apagar
window.addEventListener("pagehide", stopCamera);
