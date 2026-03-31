"""
Chat with a trained Netra model via a web UI on Modal.

Usage:
    modal serve deploy/modal_chat.py      # dev mode (hot reload, temporary URL)
    modal deploy deploy/modal_chat.py     # persistent URL
"""

import modal

app = modal.App("netra-chat")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "tokenizers", "fastapi[standard]")
    .add_local_dir("netra", remote_path="/root/netra/netra")
)

volume = modal.Volume.from_name("netra-data", create_if_missing=True)
VOLUME_PATH = "/data"


@app.cls(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root/netra")

        import torch
        import torch.nn.functional as F
        from netra import ModelConfig, Netra, NetraTokenizer

        self.torch = torch
        self.F = F

        tok_path = f"{VOLUME_PATH}/tokenizer.json"
        ckpt_path = f"{VOLUME_PATH}/checkpoints/small/final.pt"

        self.tokenizer = NetraTokenizer.from_file(tok_path)
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        config = ModelConfig(**ckpt["model_config"])
        self.model = Netra(config).to("cuda")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.n_layers = len(self.model.layers)
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict):
        prompt = request.get("prompt", "")
        max_tokens = min(request.get("max_tokens", 200), 512)
        temperature = request.get("temperature", 0.8)
        top_k = request.get("top_k", 50)
        rep_penalty = request.get("rep_penalty", 1.2)

        ids = self.torch.tensor(
            self.tokenizer.encode(prompt), dtype=self.torch.long, device="cuda"
        ).unsqueeze(0)

        with self.torch.no_grad():
            cache = [{} for _ in range(self.n_layers)]
            logits, _ = self.model(ids, cache=cache)

            for _ in range(max_tokens):
                next_logits = logits[:, -1, :]

                if rep_penalty != 1.0:
                    seen = ids[0].unique()
                    penalty = self.torch.where(next_logits[:, seen] < 0, rep_penalty, 1.0 / rep_penalty)
                    next_logits[:, seen] *= penalty

                next_logits = next_logits / max(temperature, 1e-8)

                if top_k > 0:
                    topk_vals, _ = self.torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

                probs = self.F.softmax(next_logits, dim=-1)
                next_id = self.torch.multinomial(probs, num_samples=1)
                ids = self.torch.cat([ids, next_id], dim=1)
                if next_id.item() == self.tokenizer.eot_id:
                    break
                logits, _ = self.model(next_id, cache=cache)

        text = self.tokenizer.decode(ids[0].tolist())
        return {"text": text}

    @modal.fastapi_endpoint(method="GET")
    def ui(self):
        from fastapi.responses import HTMLResponse
        return HTMLResponse(CHAT_HTML)


CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Netra Chat</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f;
    --surface: #14141f;
    --border: #1e1e2e;
    --text: #e2e2e8;
    --text-dim: #6b6b80;
    --accent: #7c6aef;
    --accent-glow: rgba(124, 106, 239, 0.15);
    --user-bg: #1a1a2e;
    --bot-bg: #0f0f1a;
  }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header h1 {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  header span {
    font-size: 12px;
    color: var(--text-dim);
    background: var(--surface);
    padding: 4px 10px;
    border-radius: 100px;
    font-family: 'JetBrains Mono', monospace;
  }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
    padding: 16px 20px;
    border-radius: 12px;
    line-height: 1.6;
    font-size: 15px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user {
    background: var(--user-bg);
    border: 1px solid var(--border);
  }
  .msg.bot {
    background: var(--bot-bg);
    border-left: 3px solid var(--accent);
  }
  .msg .label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    color: var(--text-dim);
  }
  .msg.bot .label { color: var(--accent); }
  .msg .content { font-family: 'JetBrains Mono', monospace; font-size: 14px; }
  .thinking {
    color: var(--text-dim);
    font-style: italic;
  }
  #input-area {
    padding: 16px 24px 24px;
    border-top: 1px solid var(--border);
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
    display: flex;
    gap: 10px;
  }
  #prompt {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s;
  }
  #prompt:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
  #prompt::placeholder { color: var(--text-dim); }
  button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 14px 24px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  button:hover { opacity: 0.85; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #controls {
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
    padding: 0 24px 8px;
    display: flex;
    gap: 16px;
    align-items: center;
  }
  #controls label {
    font-size: 12px;
    color: var(--text-dim);
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #controls input {
    width: 60px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    text-align: center;
    outline: none;
  }
</style>
</head>
<body>
<header>
  <h1>Netra</h1>
  <span>334M MoE &middot; small</span>
</header>

<div id="messages"></div>

<div id="controls">
  <label>temp <input type="number" id="temp" value="0.8" min="0" max="2" step="0.1"></label>
  <label>top-k <input type="number" id="top-k" value="50" min="0" max="200" step="5"></label>
  <label>rep penalty <input type="number" id="rep-pen" value="1.2" min="1.0" max="2.0" step="0.1"></label>
  <label>max tokens <input type="number" id="max-tok" value="200" min="1" max="512" step="10"></label>
</div>

<div id="input-area">
  <input type="text" id="prompt" placeholder="Say something..." autocomplete="off">
  <button id="send" onclick="send()">Generate</button>
</div>

<script>
const GENERATE_URL = window.location.origin.replace('-model-ui', '-model-generate');
const messages = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');

promptEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !sendBtn.disabled) send(); });

async function send() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  promptEl.value = '';

  addMsg('user', prompt);
  const thinkId = addMsg('bot', 'generating...', true);
  sendBtn.disabled = true;

  try {
    const res = await fetch(GENERATE_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt,
        temperature: parseFloat(document.getElementById('temp').value) || 0.8,
        top_k: parseInt(document.getElementById('top-k').value) || 50,
        rep_penalty: parseFloat(document.getElementById('rep-pen').value) || 1.2,
        max_tokens: parseInt(document.getElementById('max-tok').value) || 200,
      }),
    });
    const data = await res.json();
    updateMsg(thinkId, data.text);
  } catch (e) {
    updateMsg(thinkId, 'Error: ' + e.message);
  }
  sendBtn.disabled = false;
  promptEl.focus();
}

function addMsg(role, text, thinking) {
  const id = 'msg-' + Date.now();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.id = id;
  div.innerHTML = '<div class="label">' + (role === 'user' ? 'You' : 'Netra') + '</div>'
    + '<div class="content' + (thinking ? ' thinking' : '') + '">' + esc(text) + '</div>';
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function updateMsg(id, text) {
  const el = document.getElementById(id);
  if (el) el.querySelector('.content').className = 'content';
  if (el) el.querySelector('.content').textContent = text;
  messages.scrollTop = messages.scrollHeight;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

promptEl.focus();
</script>
</body>
</html>
"""
