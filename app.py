# app.py
import logging
import re
import json
import os
import time
import ssl
import asyncio
from typing import Dict, List, Optional, Any
from quart import Quart, request, jsonify

from agents.agent_setup import agent_manager
from services.audio_processing import handle_audio_message
from utils.message_buffer import handle_message_with_buffer, update_presence
from utils.smart_message_processor import send_message_in_chunks
from utils.conversation_manager import conversation_manager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Quart(__name__)
processed_message_ids = set()

@app.before_serving
async def startup():
    """Inicializa o agente e a base de conhecimento antes de servir requisições."""
    try:
        await agent_manager.initialize()
        logger.info("Agente e base de conhecimento inicializados com sucesso!")
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}", exc_info=True)
        raise

@app.route('/webhook', methods=['POST'])
async def webhook():
    try:
        data = await request.get_json()
        logger.debug(f"Webhook recebido: {data}")

        if not data:
            return jsonify({"status": "ignored"}), 200

        event_type = data.get("event")
        message_data = data.get("data", {})

        # Processa eventos de presença primeiro
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
            return jsonify({"status": "success", "message": "Mensagem do agente ignorada"}), 200

        # Verifica processamento duplicado
        message_id = message_data.get("key", {}).get("id")
        if message_id and message_id in processed_message_ids:
            logger.info(f"Mensagem {message_id} já processada, ignorando.")
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
            
        # Normaliza o número usando o conversation_manager
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
                task = asyncio.create_task(handle_message_with_buffer(message_text, number))
                # Define um timeout de 60 segundos
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

# Limpa o cache de mensagens periodicamente
@app.before_serving
async def setup_background_tasks():
    async def clear_message_cache():
        while True:
            await asyncio.sleep(3600)  # Limpa a cada hora
            processed_message_ids.clear()
            logger.info("Cache de mensagens processadas limpo")
    
    app.add_background_task(clear_message_cache)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)