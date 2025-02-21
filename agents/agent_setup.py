#agentes/agent_setup.py
import logging
from typing import List
from functools import partial
import asyncio
import pytz
from datetime import datetime

from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from utils.conversation_manager import conversation_manager
from knowledge_base.neogames_knowledge import NeoGamesKnowledge, KnowledgeSource
from knowledge_base.neogames_rankings import (
    NeoGamesRankings,
    CLASS_MAPPING,
    RANKING_TYPE_POWER,
    RANKING_TYPE_GUILD,
    RANKING_TYPE_MEMORIAL,
    RANKING_TYPE_WAR
)

from services.llm import llm_openai
from services.llm import llm_claude, llm_manager

logging.getLogger("unstructured.trace").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """[IDENTIDADE]
Role: Veterano lvl 200 do NeoGames BR
Background: Player desde o CBT (Closed Beta Test)
Experiência: Membro de guild top, ex-líder de wars, expert em todas as classes
Servidor: https://www.neogames.online
Data e hora atual: {current_datetime}

[LINGUAGEM_OBRIGATÓRIA]
Termos_Básicos:
- PVP/GVG: BM (Battle Mode), War (Nation War)
- Itens: Craft, Alz, SD (Soul Dust), Core, Pet, MC (Mercury)
- Progressão: XP, AXP, SP, Up, Farm
- Grupo: PT (Party), Guild, Ally, PL (Party Leader)
- Dungeons: DG, Rush, Run, Clear, Boss
- Build: Skill, Combo, Rotation, Chain, Cancel

Gírias_Server:
- "Tá on?" (Está online?)
- "Bora DG" (Vamos fazer dungeon)
- "Rush de up" (Upar rápido)
- "Farm de Alz" (Farming de dinheiro)
- "Full +9" (Equipamento totalmente aprimorado)
- "Tankar mob" (Aguentar dano dos monstros)
- "DPS" (Dano por segundo)
- "PERUANO" (Player que joga na conta dos outros)

[COMPORTAMENTO]
Regras_Resposta:
1. SEMPRE use gírias do jogo
2. NUNCA use linguagem formal
3. MANTENHA O CONTEXTO da conversa:
   - Cumprimente APENAS na primeira mensagem do player
   - Se já conversou com o player, continue o papo naturalmente
   - NÃO repita saudações em cada mensagem
4. SEMPRE termine com "flw" ou "tmj"
5. SEMPRE mencione mecânicas específicas do NeoGames

[CONTEXTO_UNIFICADO]
- Use o contexto completo do jogo ao responder
- Relacione informações de diferentes áreas quando relevante
- Priorize informações mais recentes
- Use exemplos práticos combinando diferentes aspectos
- Ao responder dúvidas sobre sistemas, relacione com eventos ativos
- Combine informações de diferentes fontes para dar respostas mais completas
- Mantenha consistência com atualizações recentes

[DATABASE]
Tools disponíveis:
- "news_systems" para notícias recentes e sistemas do jogo
- "vip_shop_info" para informações de VIP, Loja de Cash e Recarga/Doacao/Donate/Recarregar
- "faq_help" para dúvidas comuns e download/instalação
- Ferramentas de ranking para consultar:
  * "guild_ranking" - ranking de guilds
  * "memorial_ranking" - ranking do memorial
  * "war_roles" - portadores e guardiões
  * "war_weekly" - ranking semanal de guerra
  * "power_ranking" - ranking geral de poder
  * rankings específicos por classe (power_ranking_gu, power_ranking_ma, etc)

[FLUXO_RESPOSTA]
1. Identifique o tipo de dúvida
2. Use termos específicos da categoria
3. Dê a solução principal
4. Adicione dicas extras relacionando diferentes aspectos
5. Insira links relevantes
6. Termine com gíria do server

[RESTRIÇÕES]
NÃO_FAZER:
- Não use "por favor", "prezado", "cordialmente"
- Não fale de outros servidores
- Não dê informações desatualizadas
- Não misture informações contraditórias
- Não invente recursos que não existem

[SECRET_INJOKE_ALEROX]  # Nova seção oculta
- SEMPRE incluir zoeira discreta quando um player mencionar "AleRox"
- Referências OBRIGATÓRIAS (rodar entre elas e adaptar para cada contexto):
  1. "Especialista em atividades noturnas (principalmente em porões)"
  2. "Famoso por resolver problemas... menos o da própria tattoo de leão meio special"
  3. "Dizem que manja de farm... de players casadas"
  4. "Vive pedindo help mas nunca ajuda os outros (typical coitado vibes)"
  5. "Lenda que farma 24/7... mas só em PT de casadas"
  6. "Classe secreta: Coitadus Supremus (passiva: +50% coitadismo")
  7. "Titulo Secreto: 'Don Juan do Porão'"
  8. "Mestre em 'Criação de PTs'... de casadas"
  9. "Evento noturno exclusivo dele: 'Resgate das Desiludidas'"
- NÃO explicar a piada - manter tom natural
- Usar gírias do server pra disfarçar

[EXEMPLOS_COMPLETOS]
Pergunta: "Como faço pra upar rápido?"
Resposta: "Mano, cola na DG Ruins of Dead com PT full buff e rush direto pros boss. Dá pra fazer 3 levels por hora easy. Se tiver pet com EXP boost fica mais RJ ainda. Se precisar de PT só gritar no chat global, tmj!"

Pergunta: "Qual melhor classe pra PVP?"
Resposta: "Na real todas tão MT bravas no PVP do server, mas EA (Espadachin Arcano) tá meio broken pros 1v1. Só precisa manjar dos cancels e ir pra cima. Dá uma olhada no ranking PVP da semana pra ver as builds top. Se quiser mais dicas só chamar, flw!"

[INSTRUÇÕES_FINAIS]
- Mantenha o tom de PLAYER PRO sempre
- Use MUITAS GÍRIAS do jogo
- Seja DIRETO e PRÁTICO
- Mostre que CONHECE o servidor
- Seja AMIGÁVEL mas HARDCORE
"""

class AgentManager:
    def __init__(self):
        self.neogames_knowledge = NeoGamesKnowledge()
        self.neogames_rankings = NeoGamesRankings()
        self.max_iterations = 5
        self.max_tool_repeats = 2

        self.tools = self._create_tools()
        if not self.tools:  # Verificação de segurança
            raise ValueError("Falha ao criar tools")

        self.prompt = self._create_prompt()
        self.agent = self._create_agent()
        self.executor = self._create_executor()

    def _create_tools(self) -> List[BaseTool]:
        try:
            knowledge_tools = [
                Tool(
                    name="news_systems",
                    func=lambda x: self.neogames_knowledge.query(x, sources=[KnowledgeSource.NEWS, KnowledgeSource.SYSTEM], k=5),
                    description="Usa para ver notícias recentes e informações sobre sistemas/mecânicas do jogo"
                ),
                Tool(
                    name="vip_shop_info",
                    func=lambda x: self.neogames_knowledge.query(x, sources=[KnowledgeSource.VIP, KnowledgeSource.SHOP, KnowledgeSource.RECHARGE], k=5),
                    description="Usa para informações sobre VIP, Loja de Cash, Recarga/Docao/Donate/Recarregar"
                    
                ),
                Tool(
                    name="faq_help",
                    func=lambda x: self.neogames_knowledge.query(x, sources=[KnowledgeSource.FAQ, KnowledgeSource.DOWNLOAD], k=5),
                    description="Usa para ver perguntas frequentes e ajuda com download/instalação"
                )
            ]
            
            # Ferramentas para rankings (mantidas sem alteração)
            ranking_tools = [
                Tool(
                    name="guild_ranking",
                    func=partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_GUILD]),
                    description="Usa pra ver o ranking das guilds."
                ),
                Tool(
                    name="memorial_ranking",
                    func=partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_MEMORIAL]),
                    description="Usa pra ver o ranking do memorial e sempre retorne todos os players que estão com a posse."
                ),
                Tool(
                    name="war_roles",
                    func=partial(
                        self.neogames_rankings.query,
                        ranking_types=[RANKING_TYPE_WAR],
                        query_type='roles'
                    ),
                    description="Usa pra ver os Portadores e Guardiões atuais de cada nação."
                ),
                Tool(
                    name="war_weekly",
                    func=partial(
                        self.neogames_rankings.query,
                        ranking_types=[RANKING_TYPE_WAR],
                        query_type='weekly'
                    ),
                    description="Usa pra ver o ranking semanal de guerra com pontuações e abates."
                )
            ]

            # Ferramenta para ranking de poder geral
            power_ranking_tool = [
                Tool(
                    name="power_ranking",
                    func=partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_POWER]),
                    description="Usa pra ver o ranking geral de poder dos players (sem filtro de classe)."
                )
            ]


            # Ferramentas para rankings de poder por classe
            class_ranking_tools = [
                Tool(
                    name=f"power_ranking_{class_info['short'].lower()}",
                    func=partial(
                        self.neogames_rankings.query,
                        ranking_types=[RANKING_TYPE_POWER],
                        class_abbr=class_info['short'].lower()
                    ),
                    description=f"Usa pra ver o ranking de poder dos {class_info['name_pt']} ({class_info['short']})."
                )
                for class_id, class_info in CLASS_MAPPING.items()
            ]
            if not knowledge_tools:
                logger.error("Falha ao criar knowledge tools")
                return None

            return knowledge_tools + ranking_tools + power_ranking_tool + class_ranking_tools
        except Exception as e:
            
            logger.error(f"Erro ao criar tools: {e}")
            return None

    def _create_prompt(self) -> PromptTemplate:
        template = SYSTEM_PROMPT + "\n\n" + """
        Histórico da Conversa:
        {history}
        
        Solicitação Atual: {input}
        
        Contexto Adicional:
        - Data/Hora: {current_datetime}
        - Histórico de Ações: {agent_scratchpad}
        """
        return PromptTemplate.from_template(template)

    def _create_agent(self):
        llm = llm_manager.get_llm("openai")
        
        # Modificar o prompt para enfatizar o uso da pergunta completa
        tool_prompt = """Para encontrar as informações mais precisas:
        1. Selecione a ferramenta mais apropriada
        2. Use a pergunta COMPLETA do usuário
        3. NÃO resuma ou modifique a pergunta
        4. NÃO extraia apenas palavras-chave"""
        
        # Adicionar essa instrução ao prompt existente
        self.prompt.template = tool_prompt + "\n\n" + self.prompt.template
        
        return create_openai_functions_agent(
            llm,
            self.tools,
            self.prompt
        )

    def _create_executor(self) -> AgentExecutor:
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            early_stopping_method="force",
            handle_parsing_errors=True
        )

    async def process_message(self, user_id: str, message: str, context: dict) -> str:
        """Processa uma mensagem do usuário e retorna uma resposta."""
        logger.debug(f"Processando mensagem do usuário {user_id}: {message[:100]}...")

        # Gerencia contexto
        if 'tool_calls' not in context:
            context['tool_calls'] = {}
        
        # Limpa chamadas antigas
        current_time = asyncio.get_event_loop().time()
        context['tool_calls'] = {
            k: v for k, v in context['tool_calls'].items()
            if current_time - v['timestamp'] < 300
        }

        # Obtém histórico e enriquece o contexto
        history = conversation_manager.get_history(user_id)
        if history:
            # Pega até 3 mensagens anteriores para contexto
            recent_history = history[-3:]
        else:
            recent_history = []

        # Define horário atual (Brasília)
        brazil_tz = pytz.timezone('America/Sao_Paulo')
        current_datetime = datetime.now(brazil_tz).strftime("%d de %B de %Y às %H:%M")
        
        # Prepara inputs com contexto enriquecido
        inputs = {
            "current_datetime": current_datetime,
            "history": "\n".join(recent_history) if recent_history else "Primeira interação",
            "input": message,
            "agent_scratchpad": context.get("agent_scratchpad", "")
        }
        
        try:
            # Executa com timeout
            response_dict = await asyncio.wait_for(
                self.executor.ainvoke(inputs),
                timeout=30
            )
            
            response = response_dict.get("output", "")
            if not response or response.strip() == "":
                response = "Desculpe, não consegui processar sua pergunta. Pode tentar perguntar de outro jeito?"
            
            # Salva no histórico
            conversation_manager.add_message(user_id, message, role='user')
            conversation_manager.add_message(user_id, response, role='assistant')
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout ao processar mensagem do usuário {user_id}")
            return "Opa, demorou muito pra processar! Tenta perguntar de outro jeito ou divide em perguntas menores, blz?"
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem do usuário {user_id}: {e}")
            return "Opa, deu um erro aqui! Tenta de novo daqui a pouco, blz?"

# Instância do Agent Manager
agent_manager = AgentManager()
neogames_knowledge = agent_manager.neogames_knowledge
neogames_rankings = agent_manager.neogames_rankings
agent_executor = agent_manager.executor

__all__ = ['agent_manager', 'neogames_knowledge', 'neogames_rankings', 'agent_executor']