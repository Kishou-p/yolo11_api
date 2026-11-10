# YOLO11 API

API REST em **FastAPI** para carregamento, gerenciamento e inferência com modelos YOLO11, com arquitetura em camadas, validação de uploads, persistência interna de resultados e proteção das rotas sensíveis por API Key.

## Funcionalidades

- Health checks de liveness (`/health/live`) e readiness (`/health/ready`).
- Modelo carregado e aquecido automaticamente na inicialização, com verificação de SHA-256.
- Inferência por upload de imagem (JPEG, PNG, WebP) com limite de tamanho e de resolução.
- Upload original, JSON de resultado e imagem anotada salvos com escrita atômica.
- Consulta, listagem paginada e remoção de resultados por `execution_id`.
- Rotas sensíveis protegidas por `X-API-Key`, com suporte a rotação de chaves.
- Respostas sem caminhos internos; `/docs` desabilitado fora de development/test.

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
- Os arquivos em `data/uploads` e `data/outputs` não são removidos automaticamente; defina uma rotina de retenção conforme a necessidade de rastreabilidade.
