# YOLO11 API

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

API REST em **FastAPI** para carregamento, gerenciamento e inferência com modelos YOLO11.

O projeto foi desenvolvido com foco em arquitetura limpa, separação de responsabilidades, testes automatizados, proteção por API Key, persistência de resultados e execução com Docker.

---

## Visão geral

A YOLO11 API permite enviar uma imagem para inferência com um modelo YOLO11.

Fluxo principal:

```text
Imagem enviada
    ↓
API valida o upload
    ↓
Modelo YOLO11 executa a inferência
    ↓
API salva o upload, o JSON de resultado e a imagem anotada
    ↓
API retorna execution_id, result_url e annotated_image_url
```

A resposta pública não expõe caminhos internos do servidor.

---

## Status do projeto

Estado atual validado:

- API base funcionando.
- Arquitetura limpa em camadas.
- YOLO11 isolado em `app/services/yolo_service.py`.
- Modelo cacheado em memória via `app.state`.
- Inferência com upload de imagem funcionando.
- Upload original, JSON de resultado e imagem anotada salvos internamente.
- Resposta pública sem vazamento de paths internos.
- Rotas sensíveis protegidas por `X-API-Key`.
- CORS controlado por configuração.
- Testes automatizados passando.
- Docker validado.
- Docker Compose validado com volumes persistentes.
- GitHub Actions configurado para testes e Docker build.

---

## Arquitetura

```text
app/
├── core/          # Configuração, dependências, logging e app state
├── exceptions/    # Exceções customizadas e handlers globais
├── routes/        # Camada HTTP
├── schemas/       # Contratos Pydantic
├── services/      # Regras de negócio e orquestração
├── utils/         # Funções auxiliares puras
└── main.py        # Factory da aplicação FastAPI

data/
├── models/        # Modelos YOLO
├── uploads/       # Imagens recebidas
└── outputs/       # JSONs e imagens anotadas

tests/             # Testes automatizados
```

Regra principal do projeto:

```text
ultralytics só deve ser importado em app/services/yolo_service.py
```

As rotas não devem importar YOLO diretamente, manipular regra de negócio pesada ou acessar caminhos internos.

---

## Endpoints

### Público

```http
GET /api/v1/health
```

### Protegidos por API Key

As rotas abaixo exigem o header:

```http
X-API-Key: dev-secret-key
```

```http
GET  /api/v1/models/status
POST /api/v1/models/load
POST /api/v1/inference/image
GET  /api/v1/results/{execution_id}
GET  /api/v1/results/{execution_id}/image
```

---

## Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
APP_NAME=YOLO11 API
APP_VERSION=0.1.0
ENVIRONMENT=development
LOG_LEVEL=INFO
API_PREFIX=/api/v1

YOLO_MODEL_PATH=data/models/yolo11n.pt
MAX_IMAGE_SIZE_MB=10

UPLOAD_DIR=data/uploads
OUTPUT_DIR=data/outputs

API_KEY=dev-secret-key

ALLOWED_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000","http://localhost:8000","http://127.0.0.1:8000"]
```

Ao usar Docker, evite aspas em valores simples.

Use:

```env
LOG_LEVEL=INFO
API_KEY=dev-secret-key
API_PREFIX=/api/v1
```

Evite:

```env
LOG_LEVEL="INFO"
API_KEY="dev-secret-key"
API_PREFIX="/api/v1"
```

---

## Rodando localmente

Acesse o projeto:

```powershell
cd C:\Projetos\yolo11_api
```

Ative o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Rode os testes:

```powershell
python -m pytest
```

Suba a API:

```powershell
uvicorn app.main:create_app --factory --reload
```

Acesse a documentação Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Rodando com Docker

Build da imagem:

```powershell
docker build -t yolo11-api:latest .
```

Rodar container:

```powershell
docker run --rm `
  -p 8000:8000 `
  --env-file .env `
  -e YOLO_CONFIG_DIR=/tmp `
  yolo11-api:latest
```

Testar health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

---

## Rodando com Docker Compose

Subir a API:

```powershell
docker compose up -d --build
```

Ver containers:

```powershell
docker compose ps
```

Ver logs:

```powershell
docker compose logs -f
```

Parar a API:

```powershell
docker compose down
```

O `docker-compose.yml` usa volumes persistentes para:

- `data/uploads`
- `data/outputs`
- `data/models`

---

## Demo rápida

Com a API rodando via Docker Compose:

```powershell
docker compose up -d --build
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

Carregar modelo:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/models/load `
  -Headers @{ "X-API-Key" = "dev-secret-key" }
```

Executar inferência:

```powershell
$imagePath = "C:\caminho\para\sua-imagem.png"

curl.exe -X POST `
  "http://127.0.0.1:8000/api/v1/inference/image" `
  -H "X-API-Key: dev-secret-key" `
  -F "file=@$imagePath"
```

Exemplo de resposta:

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

Consultar resultado:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/v1/results/1a3468eee54747699873d461cf15db31" `
  -Headers @{ "X-API-Key" = "dev-secret-key" }
```

Baixar imagem anotada:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/api/v1/results/1a3468eee54747699873d461cf15db31/image" `
  -Headers @{ "X-API-Key" = "dev-secret-key" } `
  -OutFile ".\annotated_result.png"
```

---

## Testes

Rodar todos os testes:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Estado validado:

```text
87 passed
```

---

## CI

O projeto usa GitHub Actions para validar:

- instalação de dependências Python;
- execução dos testes automatizados;
- build da imagem Docker.

O workflow está em:

```text
.github/workflows/ci.yml
```

---

## Segurança

A API usa proteção por API Key nas rotas sensíveis.

Header esperado:

```http
X-API-Key: dev-secret-key
```

Para uso real, altere a variável:

```env
API_KEY=dev-secret-key
```

para uma chave segura.

A autenticação atual é suficiente para demonstração e portfólio, mas para produção real recomenda-se evoluir para múltiplas chaves, escopos de permissão ou JWT.

---

## Persistência

A API salva internamente:

- imagem original enviada;
- JSON com resultado da inferência;
- imagem anotada.

A resposta pública não expõe paths internos. Em vez disso, usa:

- `execution_id`
- `result_url`
- `annotated_image_url`

---

## Auditoria técnica

O projeto passou por uma auditoria técnica cobrindo:

- arquitetura;
- separação de responsabilidades;
- segurança básica;
- Docker e Docker Compose;
- persistência de resultados;
- testes automatizados;
- riscos técnicos;
- roadmap de evolução.

Principais pontos de atenção:

- A imagem Docker ainda é grande por causa de dependências como `ultralytics`, `torch` e OpenCV.
- A API Key única é adequada para demonstração, mas limitada para múltiplos usuários.
- O cache do modelo em `app.state` funciona bem em um processo, mas precisa de adaptação para múltiplos workers.
- O projeto pode evoluir com histórico de resultados, deleção de resultados, logs estruturados e status de storage.

---

## Roadmap

Próximos passos planejados:

1. Consolidar documentação e repositório.
2. Adicionar endpoint para listar resultados.
3. Adicionar endpoint para deletar resultados.
4. Melhorar logs com `execution_id`.
5. Adicionar endpoint de status do storage.
6. Otimizar tamanho da imagem Docker.
7. Criar interface web para upload e visualização.
8. Criar fluxo de feedback para dataset.
9. Evoluir para detecção de anomalias.
10. Automatizar coleta e organização de imagens.

---

## Licença

Este projeto está licenciado sob a licença MIT.
