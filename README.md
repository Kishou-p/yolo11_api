\# YOLO11 API

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

API REST em \*\*FastAPI\*\* para carregamento, gerenciamento e inferência com modelos YOLO11, seguindo arquitetura limpa em camadas, validação de uploads, persistência interna de resultados e proteção de rotas sensíveis por API Key.



\## Demo rápida com Docker Compose

Subir a API:

```powershell
docker compose up -d --build


\## Status do projeto



Estado atual:



\- API base funcionando.

\- Arquitetura limpa em camadas.

\- YOLO11 isolado em `app/services/yolo\_service.py`.

\- Cache de modelo em memória via `app.state`.

\- Inferência com upload de imagem funcionando.

\- Upload original, JSON de resultado e imagem anotada salvos internamente.

\- Resposta pública sem vazamento de paths internos.

\- Rotas sensíveis protegidas por `X-API-Key`.

\- CORS controlado por configuração.

\- Testes automatizados previamente validados com `87 passed`.

\- Docker build validado.

\- Docker run validado.

\- Docker Compose validado com volumes persistentes.



\## Objetivo



Este projeto fornece uma API para executar inferência com YOLO11 de forma organizada, testável e segura.



A API permite:



\- Verificar saúde da aplicação.

\- Verificar status do modelo.

\- Carregar modelo YOLO11 em memória.

\- Enviar imagem para inferência.

\- Salvar artefatos internos da execução.

\- Consultar resultado por `execution\_id`.

\- Consultar imagem anotada por `execution\_id`.



\---



\## Arquitetura



```text

app/

├── core/          # Configuração, dependências, logging e app state

├── exceptions/    # Exceções customizadas e handlers globais

├── routes/        # Camada HTTP apenas

├── schemas/       # Contratos Pydantic de entrada e saída

├── services/      # Regras de negócio e orquestração

├── utils/         # Funções auxiliares puras

└── main.py        # Factory da aplicação FastAPI



data/

├── models/        # Modelos YOLO

├── uploads/       # Imagens recebidas

└── outputs/       # JSONs e imagens anotadas



tests/             # Testes automatizados

