import logging
from typing import List
from functools import partial
import asyncio

from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from knowledge_base.neogames_knowledge import NeoGamesKnowledge, KnowledgeSource
from knowledge_base.neogames_rankings import (
    NeoGamesRankings,
    CLASS_MAPPING,
    RANKING_TYPE_POWER,
    RANKING_TYPE_GUILD,
    RANKING_TYPE_MEMORIAL
)

from services.llm import llm_openai

logging.getLogger("unstructured.trace").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """# Quem você é
Você é o assistente do NeoGames, um servidor BR de Cabal Online (https://www.neogames.online).
Você é um player veterano e conhece tudo sobre o jogo e o servidor. 
Você fala de forma casual e usa gírias comuns dos players.

# Seu jeito de ser
- Fale como um player mesmo, nada de formalidades
- Use gírias do jogo (tipo: up, farm, mob, drop, etc)
- Seja animado e empolgado
- Mostre que manja do Cabal e do servidor
- Ajude os players como se fosse um amigo

# Como responder
1. Vá direto ao ponto que o player quer saber
2. Passe os links importantes se precisar
3. Dê dicas extras se tiver
4. Termine com algo tipo "qualquer coisa tamo aí" ou "se precisar é só chamar"

# O que você sabe
- Tudo sobre o servidor e o jogo
- Notícias e updates mais recentes
- Como baixar e instalar
- FAQs e dúvidas comuns
- Rankings (power, war, guild)
- Todos os sistemas especiais

# Como ajudar
- Priorize ajudar com acesso, download e pagamentos
- Pra problemas mais complexos, mande falar com o suporte
- Sempre cheque as infos antes de responder

# Suas ferramentas
- *game_info* pra info geral
- *news_info* pra notícias
- *download_info* pra download
- *faq_info* pra dúvidas comuns
- *power_ranking* pra ranking de poder (geral)
- *power_ranking_gu* pra ranking de Guerreiros
- *power_ranking_du* pra ranking de Duelistas
- *power_ranking_ma* pra ranking de Magos
- *power_ranking_aa* pra ranking de Arqueiros Arcanos
- *power_ranking_ga* pra ranking de Guardiões Arcanos
- *power_ranking_ea* pra ranking de Espadachins Arcanos
- *power_ranking_gl* pra ranking de Gladiadores
- *power_ranking_at* pra ranking de Atiradores
- *power_ranking_mn* pra ranking de Magos Negros
- *guild_ranking* pra ranking de guild
- *memorial_ranking* pra ranking do memorial
- *system_info* pra sistemas especiais
- *vip_info* pra benefícios VIP
- *recharge_info* pra recargas
- *shop_info* pra loja do servidor

Lembra: você é um player ajudando outro player. Mantenha o papo informal e descontraído!
"""

class AgentManager:
    def __init__(self):
        self.neogames_knowledge = NeoGamesKnowledge()
        self.neogames_rankings = NeoGamesRankings()
        self.max_iterations = 8  # Limite de iterações por consulta
        self.max_tool_repeats = 2  # Limite de repetições da mesma ferramenta

        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.agent = self._create_agent()
        self.executor = self._create_executor()
        
    def _create_tools(self) -> List[BaseTool]:
        def wrap_tool_query(func, tool_name):
            """Wrapper para adicionar controle e tratamento de erros nas queries"""
            def wrapped_query(*args, **kwargs):
                try:
                    # Verifica se a função é uma coroutine
                    if asyncio.iscoroutinefunction(func):
                        # Cria um evento loop se não existir
                        loop = asyncio.get_event_loop()
                        result = loop.run_until_complete(func(*args, **kwargs))
                    else:
                        result = func(*args, **kwargs)

                    if not result or result.strip() == "":
                        return f"Não encontrei informações para sua pergunta sobre {tool_name}. Tente ser mais específico ou pergunte de outra forma."
                    return result
                except Exception as e:
                    logger.error(f"Erro na ferramenta {tool_name}: {e}")
                    return f"Desculpe, tive um problema ao buscar as informações. Tente novamente."
            return wrapped_query

        # Ferramentas para conteúdo geral
        general_tools = [
            Tool(
                name="game_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.MAIN]),
                    "game_info"
                ),
                description="Usa para info geral do servidor NeoGames e sobre o jogo."
            ),
            Tool(
                name="news_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.NEWS]),
                    "news_info"
                ),
                description="Usa pra ver as últimas notícias e atualizações do servidor."
            ),
            Tool(
                name="download_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.DOWNLOAD]),
                    "download_info"
                ),
                description="Usa pra info de download e instalação do jogo."
            ),
            Tool(
                name="faq_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.FAQ]),
                    "faq_info"
                ),
                description="Usa pra ver as perguntas mais comuns e respostas."
            ),
            Tool(
                name="vip_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.VIP]),
                    "vip_info"
                ),
                description="Usa pra ver os benefícios VIP e pacotes premium."
            ),
            Tool(
                name="recharge_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.RECHARGE]),
                    "recharge_info"
                ),
                description="Usa pra ver como fazer recargas e formas de pagamento."
            ),
            Tool(
                name="shop_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.SHOP]),
                    "shop_info"
                ),
                description="Usa pra ver a loja e os itens disponíveis."
            ),
            Tool(
                name="system_info",
                func=wrap_tool_query(
                    partial(self.neogames_knowledge.query, sources=[KnowledgeSource.SYSTEM]),
                    "system_info"
                ),
                description="Usa pra ver os sistemas especiais do servidor."
            )
        ]

        # Ferramentas para rankings
        ranking_tools = [
            Tool(
                name="guild_ranking",
                func=wrap_tool_query(
                    partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_GUILD]),
                    "guild_ranking"
                ),
                description="Usa pra ver o ranking das guilds."
            ),
            Tool(
                name="memorial_ranking",
                func=wrap_tool_query(
                    partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_MEMORIAL]),
                    "memorial_ranking"
                ),
                description="Usa pra ver o ranking do memorial e sempre retorne todos os players que estão com a posse."
            )
        ]

        # Ferramenta para ranking de poder geral
        power_ranking_tool = [
            Tool(
                name="power_ranking",
                func=wrap_tool_query(
                    partial(self.neogames_rankings.query, ranking_types=[RANKING_TYPE_POWER]),
                    "power_ranking"
                ),
                description="Usa pra ver o ranking geral de poder dos players (sem filtro de classe)."
            )
        ]

        # Ferramentas para rankings de poder por classe
        class_ranking_tools = [
            Tool(
                name=f"power_ranking_{class_info['short'].lower()}",
                func=wrap_tool_query(
                    partial(
                        self.neogames_rankings.query,
                        ranking_types=[RANKING_TYPE_POWER],
                        class_abbr=class_info['short'].lower()
                    ),
                    f"power_ranking_{class_info['short'].lower()}"
                ),
                description=f"Usa pra ver o ranking de poder dos {class_info['name_pt']} ({class_info['short']})."
            )
            for class_id, class_info in CLASS_MAPPING.items()
        ]

        # Combina todas as ferramentas
        return general_tools + ranking_tools + power_ranking_tool + class_ranking_tools

    def _create_prompt(self) -> PromptTemplate:
        template = SYSTEM_PROMPT + "\n\n" + (
            "Histórico da Conversa:\n{history}\n\n"
            "Solicitação Atual: {input}\n\n"
            "Histórico de Ações:\n{agent_scratchpad}\n"
        )
        return PromptTemplate.from_template(template)

    def _create_agent(self):
        return create_openai_functions_agent(
            llm_openai,
            self.tools,
            self.prompt
        )

    def _create_executor(self) -> AgentExecutor:
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            early_stopping_method="force"
        )

    async def initialize(self):
        """Inicializa as duas bases de conhecimento."""
        try:
            logger.info("Iniciando inicialização da base de conhecimento...")
            await self.neogames_knowledge.initialize()
            logger.info("Base de conhecimento inicializada com sucesso")
            
            logger.info("Iniciando inicialização da base de rankings...")
            await self.neogames_rankings.initialize()
            logger.info("Base de rankings inicializada com sucesso")
            
            logger.info("Todas as bases inicializadas com sucesso")
        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")
            raise

    async def process_message(self, user_id: str, message: str, context: dict) -> str:
        """
        Processa uma mensagem do usuário e retorna uma resposta.
        
        Args:
            user_id: ID do usuário que enviou a mensagem
            message: Texto da mensagem
            context: Dicionário com o contexto da conversa
            
        Returns:
            str: Resposta para o usuário
        """
        logger.debug(f"Processando mensagem do usuário {user_id}: {message[:100]}...")
        
        # Inicializa ou atualiza controle de ferramentas no contexto
        if 'tool_calls' not in context:
            context['tool_calls'] = {}
        
        # Limpa chamadas antigas (mais de 5 minutos)
        current_time = asyncio.get_event_loop().time()
        context['tool_calls'] = {
            k: v for k, v in context['tool_calls'].items()
            if current_time - v['timestamp'] < 300
        }
        
        inputs = {
            "history": context.get("history", "") or "Nenhum histórico",
            "input": message,
            "agent_scratchpad": context.get("agent_scratchpad", "") or ""
        }
        
        try:
            response_dict = await asyncio.wait_for(
                self.executor.ainvoke(inputs),
                timeout=30
            )
            
            response = response_dict.get("output", "")
            if not response or response.strip() == "":
                return "Desculpe, não consegui processar sua pergunta. Pode tentar perguntar de outro jeito?"
                
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