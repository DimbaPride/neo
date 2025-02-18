import logging
import re
import json
import os
import time
import ssl
import asyncio
import pytz
from datetime import datetime
from typing import Dict, List, Optional, Any
from quart import Quart, request, jsonify
from hypercorn.config import Config

# Importa o Agent Manager já atualizado (que contém a base de conhecimento)
from agents.agent_setup import agent_manager
from services.audio_processing import handle_audio_message
from utils.message_buffer import handle_message_with_buffer, update_presence
from utils.smart_message_processor import send_message_in_chunks
from utils.conversation_manager import conversation_manager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializa a aplicação Quart
app = Quart(__name__)

# Configuração do Hypercorn com timeout reduzido
config = Config()
config.startup_timeout = 30  # 30 segundos para startup inicial
config.keep_alive_timeout = 300

# Variáveis globais
processed_message_ids = set()
initialization_task = None

def get_brazil_time() -> str:
    """Retorna a data e hora atual no fuso horário de Brasília."""
    brazil_tz = pytz.timezone('America/Sao_Paulo')
    return datetime.now(brazil_tz).strftime("%Y-%m-%d %H:%M:%S")

@app.before_serving
async def startup():
    """
    Função executada antes do servidor começar a servir requisições.
    Inicia o processo de inicialização em background.
    """
    global initialization_task
    
    async def initialize_background():
        try:
            logger.info("Iniciando inicialização em background...")
            # Inicializa a base de conhecimento primeiro
            await agent_manager.neogames_knowledge.initialize()
            logger.info("Base de conhecimento inicializada")
            
            await asyncio.sleep(2)  # Pequena pausa
            
            # Depois inicializa os rankings
            await agent_manager.neogames_rankings.initialize()
            logger.info("Base de rankings inicializada")
            
            # Configura a limpeza periódica do cache
            async def clear_message_cache():
                while True:
                    try:
                        await asyncio.sleep(3600)  # Limpa a cada hora
                        processed_message_ids.clear()
                        logger.info("Cache de mensagens processadas limpo")
                    except Exception as e:
                        logger.error(f"Erro ao limpar cache: {e}")
                        await asyncio.sleep(60)
            
            app.add_background_task(clear_message_cache)
            logger.info("Inicialização em background concluída!")
            
        except Exception as e:
            logger.error(f"Erro durante inicialização em background: {e}")

    # Inicia a tarefa em background sem aguardar
    initialization_task = asyncio.create_task(initialize_background())
    logger.info("Servidor iniciando - inicialização continuará em background")

@app.after_serving
async def shutdown():
    """Função executada quando o servidor é desligado."""
    try:
        logger.info("Iniciando desligamento dos serviços...")
        
        # Cancela a tarefa de inicialização se ainda estiver rodando
        if initialization_task:
            initialization_task.cancel()
            try:
                await initialization_task
            except asyncio.CancelledError:
                pass
        
        # Desliga os serviços
        if hasattr(agent_manager.neogames_knowledge, 'shutdown'):
            await agent_manager.neogames_knowledge.shutdown()
        logger.info("Serviços desligados com sucesso!")
    except Exception as e:
        logger.error(f"Erro no desligamento: {str(e)}")

async def process_user_message(message: str, number: str):
    """
    Processa a mensagem do usuário com o contexto atual.
    """
    # Verifica se a inicialização está completa
    if initialization_task and not initialization_task.done():
        await send_message_in_chunks(
            "O sistema ainda está inicializando. Por favor, aguarde alguns minutos.",
            number
        )
        return

    try:
        # Monta o contexto para a mensagem
        user_context = {
            'current_datetime': get_brazil_time(),
            'current_user': number,
        }
        
        # Processa a mensagem usando o agent_manager
        response = await agent_manager.process_message(
            user_id=number,
            message=message,
            context=user_context
        )
        
        # Envia a resposta usando o processador de mensagens
        await send_message_in_chunks(response, number)
        
    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}", exc_info=True)
        raise

@app.route('/webhook', methods=['POST'])
async def webhook():
    """Endpoint principal para receber webhooks."""
    try:
        # Verifica se o sistema ainda está inicializando
        if initialization_task and not initialization_task.done():
            return jsonify({
                "status": "initializing",
                "message": "O sistema ainda está inicializando. Por favor, aguarde alguns minutos."
            }), 503

        data = await request.get_json()
        logger.debug(f"Webhook recebido: {data}")

        if not data:
            return jsonify({"status": "ignored"}), 200

        event_type = data.get("event")
        message_data = data.get("data", {})

        # Processa eventos de presença
        if event_type == "presence.update":
            try:
                presence_data = message_data.get("presences", {})
                for number, status in presence_data.items():
                    number = number.split("@")[0]
                    logger.debug(f"Atualizando presença para {number}: {status}")
                    update_presence(number, status)
                return jsonify({"status": "success"}), 200
            except Exception as e:
                logger.error(f"Erro ao processar presence.update: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

        if isinstance(message_data, list) and message_data:
            message_data = message_data[0]

        # Verifica se a mensagem foi enviada pelo próprio agente
        agent_number = conversation_manager.normalize_phone('5511911043825')
        if message_data.get('sender') == f"{agent_number}@s.whatsapp.net":
            logger.info("Mensagem enviada pelo agente, ignorando...")
            return jsonify({"status": "success"}), 200

        # Verifica processamento duplicado
        message_id = message_data.get("key", {}).get("id")
        if message_id and message_id in processed_message_ids:
            logger.info(f"Mensagem {message_id} já processada.")
            return jsonify({"status": "ignored"}), 200

        # Extrai e normaliza o número do remetente
        remote_jid = (
            message_data.get("key", {}).get("remoteJid", "")
            or message_data.get("remoteJid", "")
            or message_data.get("jid", "")
        )
        raw_number = remote_jid.split("@")[0].split(":")[0] if remote_jid else ""
        if not raw_number:
            return jsonify({"status": "ignored"}), 200
            
        number = conversation_manager.normalize_phone(raw_number)

        if event_type == "messages.upsert":
            msg_content = message_data.get("message", {})

            # Processa mensagem de áudio
            if "audioMessage" in msg_content:
                base64_data = msg_content.get("base64") or message_data.get("base64")
                if base64_data:
                    await handle_audio_message({"base64": base64_data}, number)
                    processed_message_ids.add(message_id)
                    return jsonify({"status": "processed"}), 200
                return jsonify({"status": "error", "message": "Base64 não encontrado"}), 200

            # Processa mensagem de texto
            message_text = (
                msg_content.get("conversation")
                or msg_content.get("extendedTextMessage", {}).get("text")
            )
            if message_text:
                task = asyncio.create_task(process_user_message(message_text, number))
                try:
                    await asyncio.wait_for(task, timeout=60.0)
                    processed_message_ids.add(message_id)
                    return jsonify({"status": "processed"}), 200
                except asyncio.TimeoutError:
                    logger.error("Timeout ao processar mensagem")
                    return jsonify({"status": "error", "message": "Timeout"}), 500

        return jsonify({"status": "ignored"}), 200

    except Exception as e:
        logger.error(f"Erro no webhook: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
async def health_check():
    """Endpoint para verificar o status do servidor."""
    return jsonify({
        "status": "initializing" if (initialization_task and not initialization_task.done()) else "ready",
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Erro ao iniciar o servidor: {e}", exc_info=True)