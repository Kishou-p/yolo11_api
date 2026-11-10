# YOLO11 API

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

API REST em **FastAPI** para carregamento, gerenciamento e inferência com modelos YOLO11, com arquitetura em camadas, validação de uploads, persistência interna de resultados e proteção das rotas sensíveis por API Key.

## Visão geral

```text
Imagem enviada
    ↓
API valida o upload (tipo, assinatura, tamanho e resolução)
    ↓
Modelo YOLO11 executa a inferência
    ↓
API salva o upload, o JSON de resultado e a imagem anotada
    ↓
API retorna execution_id, result_url e annotated_image_url
```

## Funcionalidades

- Health checks de liveness (`/health/live`) e readiness (`/health/ready`).
- Modelo carregado e aquecido automaticamente na inicialização, com verificação de SHA-256.
- Inferência por upload de imagem (JPEG, PNG, WebP) com limite de tamanho e de resolução.
- Upload original, JSON de resultado e imagem anotada salvos com escrita atômica.
- Consulta, listagem paginada e remoção de resultados por `execution_id`.
- Rotas sensíveis protegidas por `X-API-Key`, com suporte a rotação de chaves.
- Respostas sem caminhos internos; `/docs` desabilitado fora de development/test.
- Parâmetros de inferência configuráveis; cada resultado registra modelo (nome e SHA-256), parâmetros, data e tempos.
- Retenção automática de arquivos e opção de não salvar o upload ou a imagem anotada.
- Logs em texto ou JSON com `request_id`, e métricas Prometheus em `/metrics`.

## Arquitetura

```text
app/
├── core/          # Configuração, dependências e logging
├── exceptions/    # Exceções customizadas e handlers globais
├── routes/        # Camada HTTP
├── schemas/       # Contratos Pydantic de entrada e saída
├── services/      # Regras de negócio e orquestração
├── utils/         # Funções auxiliares
└── main.py        # Factory da aplicação e ciclo de vida (carga do modelo)

data/
├── models/        # Modelos YOLO (montado somente leitura no container)
├── uploads/       # Imagens recebidas
└── outputs/       # JSONs e imagens anotadas

tests/             # Testes automatizados
```

Regra principal do projeto: `ultralytics` só deve ser importado em `app/services/yolo_service.py`. As rotas não importam YOLO diretamente, não fazem regra de negócio pesada e não acessam caminhos internos.

## Endpoints

Prefixo padrão: `/api/v1`.

| Método | Rota | Autenticação | Descrição |
| --- | --- | --- | --- |
| GET | `/health`, `/health/live` | Não | Processo está vivo |
| GET | `/health/ready` | Não | Modelo carregado, diretórios graváveis e espaço em disco (503 se não estiver pronto) |
| GET | `/models/status` | Sim | Nome do modelo, existência e se está carregado |
| POST | `/models/load` | Sim | Carrega o modelo (útil se a carga na inicialização falhou) |
| POST | `/inference/image` | Sim | Envia imagem (`multipart/form-data`, campo `file`) |
| GET | `/results?limit=100&offset=0` | Sim | Lista resultados, do mais recente para o mais antigo |
| GET | `/results/{execution_id}` | Sim | JSON do resultado |
| GET | `/results/{execution_id}/image` | Sim | Imagem anotada |
| DELETE | `/results/{execution_id}` | Sim | Remove resultado, imagem anotada e upload |
| GET | `/metrics` (sem prefixo) | Sim | Métricas Prometheus (desligue com `METRICS_ENABLED=false`) |

Toda resposta traz o header `X-Request-ID`. Se o cliente enviar um `X-Request-ID` válido (até 64 caracteres, letras, números, `.`, `_` e `-`), ele é reaproveitado, o que permite rastrear uma peça ou câmera de ponta a ponta nos logs.

## Configuração

Copie `.env.example` para `.env` e ajuste. Principais variáveis:

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `ENVIRONMENT` | `production` | `development`, `test`, `staging` ou `production`. Fora de development/test exige chave forte e desliga `/docs` |
| `API_KEY` | — | Chave principal. Mínimo de 32 caracteres fora de development/test |
| `EXTRA_API_KEYS` | `[]` | Chaves adicionais aceitas (JSON), para rotação sem downtime |
| `YOLO_MODEL_PATH` | `data/models/yolo11n.pt` | Caminho do modelo |
| `YOLO_MODEL_SHA256` | — | Hash esperado do modelo; a carga é recusada se não bater |
| `LOAD_MODEL_ON_STARTUP` | `true` | Carrega e aquece o modelo ao iniciar |
| `MAX_IMAGE_SIZE_MB` | `10` | Tamanho máximo do upload |
| `MAX_IMAGE_PIXELS` | `40000000` | Resolução máxima (largura × altura) |
| `MIN_FREE_DISK_MB` | `500` | Abaixo disso, `/health/ready` retorna 503 |
| `ALLOWED_ORIGINS` | localhost:3000 | Origens CORS permitidas (JSON) |
| `YOLO_CONF` / `YOLO_IOU` | `0.25` / `0.7` | Confiança mínima e IoU do NMS |
| `YOLO_IMGSZ` | `640` | Tamanho de entrada da rede |
| `YOLO_MAX_DET` | `300` | Máximo de detecções por imagem |
| `YOLO_DEVICE` | automático | `cpu`, `0` (primeira GPU) etc. |
| `YOLO_HALF` | `false` | Meia precisão (só GPU) |
| `SAVE_UPLOADS` / `SAVE_ANNOTATED` | `true` | Salvar a imagem original e a anotada |
| `ANNOTATED_IMAGE_FORMAT` | `png` | `png` ou `jpg` (bem menor) |
| `RETENTION_DAYS` | `0` | Apaga arquivos mais antigos que N dias (0 = desligado) |
| `RETENTION_INTERVAL_MINUTES` | `60` | Intervalo da limpeza |
| `LOG_FORMAT` | `text` | `text` ou `json` |
| `METRICS_ENABLED` | `true` | Habilita `GET /metrics` |

Gerar uma chave:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn app.main:create_app --factory --reload
```

## Demo rápida (PowerShell)

Com a API rodando, o modelo já é carregado na inicialização:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health/ready

$headers = @{ "X-API-Key" = "<sua API_KEY>" }

curl.exe -X POST "http://127.0.0.1:8000/api/v1/inference/image" `
  -H "X-API-Key: <sua API_KEY>" `
  -F "file=@C:\caminho\para\imagem.png"

Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/results/<execution_id>" -Headers $headers

Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/results/<execution_id>/image" `
  -Headers $headers -OutFile ".\annotated_result.png"
```

Exemplo de resposta da inferência:

```json
{
  "execution_id": "1a3468eee54747699873d461cf15db31",
  "filename": "image.png",
  "result_url": "/api/v1/results/1a3468eee54747699873d461cf15db31",
  "annotated_image_url": "/api/v1/results/1a3468eee54747699873d461cf15db31/image",
  "detections_count": 0,
  "detections": []
}
```

No `.env`, evite aspas em valores simples (`API_KEY=abc`, não `API_KEY="abc"`), porque o Docker as mantém no valor.

## Docker

```bash
cp .env.example .env   # ajuste API_KEY e ENVIRONMENT
# O container roda como o usuário 10001: dê permissão de escrita nos volumes
sudo chown -R 10001:10001 data/uploads data/outputs
docker compose up -d --build
docker compose ps        # o status mostra "healthy" quando /health/ready responde 200
```

A imagem instala o PyTorch para CPU por padrão. Para GPU NVIDIA:

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

e adicione a reserva de GPU (`deploy.resources.reservations.devices`) no `docker-compose.yml`.

## Produção

- A porta 8000 é publicada apenas em `127.0.0.1`. Coloque um proxy reverso com TLS (Nginx, Traefik ou Caddy) na frente, com `client_max_body_size` próximo de `MAX_IMAGE_SIZE_MB` e limite de requisições.
- Mantenha a API em uma rede segmentada, acessível só pelos sistemas que precisam dela.
- Para trocar o modelo: copie o novo arquivo para `data/models/`, atualize `YOLO_MODEL_SHA256` e reinicie o container.
- Defina `RETENTION_DAYS` conforme a necessidade de rastreabilidade; com `0`, os arquivos em `data/uploads` e `data/outputs` nunca são apagados.
- Use `LOG_FORMAT=json` para enviar logs a Loki, ELK ou Graylog.

## Observabilidade

Exemplo de coleta pelo Prometheus (2.55+ ou 3.x, que aceitam headers customizados):

```yaml
scrape_configs:
  - job_name: yolo11-api
    metrics_path: /metrics
    static_configs:
      - targets: ["yolo11-api:8000"]
    http_headers:
      X-API-Key:
        secrets: ["<sua API_KEY>"]
```

Principais métricas:

- `inference_total{outcome}`: inferências por resultado (`success`, `rejected` ou `error`).
- `inference_stage_duration_seconds{stage}`: tempo de decode, de inferência e total.
- `inference_detections`: detecções por imagem.
- `http_requests_total` e `http_request_duration_seconds`: por método, rota e status.
- `model_loaded`: 1 quando o modelo está carregado.

## CI

O workflow em `.github/workflows/ci.yml` roda os testes automatizados e o build da imagem Docker a cada push na `main` e em cada pull request.

## Roadmap

1. Interface web para upload e visualização.
2. Fluxo de feedback para dataset.
3. Evolução para detecção de anomalias.
4. Coleta e organização automática de imagens.

## Licença

Este projeto está licenciado sob a licença MIT.
