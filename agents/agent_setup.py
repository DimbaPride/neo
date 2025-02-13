# agent_setup.py
import logging
from typing import List
from functools import partial
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from knowledge_base.knowledge_system import CabalKnowledgeSystem, KnowledgeSource

logger = logging.getLogger(__name__)

class AgentManager:
    """Gerencia a criação e configuração do agente de suporte do Cabal."""
    
    def __init__(self):
        self.knowledge_system = CabalKnowledgeSystem()
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.llm = ChatOpenAI(temperature=0.3)
        self.agent = self._create_agent()
        self.executor = self._create_executor()

    async def neo_knowledge(self, question: str) -> str:
        """Consulta informações específicas do servidor Cabal NEO"""
        response = await self.knowledge_system.query(question, sources=[KnowledgeSource.NEO_CABAL])
        return response

    async def game_knowledge(self, question: str) -> str:
        """Consulta informações gerais sobre o jogo Cabal Online"""
        response = await self.knowledge_system.query(question, sources=[KnowledgeSource.MRWORMY])
        return response

    async def combined_knowledge(self, question: str) -> str:
        """Consulta todas as fontes de conhecimento disponíveis"""
        response = await self.knowledge_system.query(question)
        return response

    def _create_tools(self) -> List[BaseTool]:
        """Cria e retorna a lista de ferramentas disponíveis para o agente."""
        return [
            Tool(
                name="neo_knowledge",
                func=self.neo_knowledge,
                description="Consulta informações específicas do servidor Cabal NEO. Use esta ferramenta para responder perguntas sobre o servidor, cash, rankings, eventos atuais e informações específicas do NEO."
            ),
            Tool(
                name="game_knowledge",
                func=self.game_knowledge,
                description="Consulta informações gerais sobre o jogo Cabal Online. Use esta ferramenta para responder perguntas sobre mecânicas do jogo, classes, dungeons, itens, skills e sistemas do jogo."
            ),
            Tool(
                name="combined_knowledge",
                func=self.combined_knowledge,
                description="Consulta todas as fontes de conhecimento disponíveis. Use quando precisar de uma visão completa ou quando não tiver certeza de qual fonte usar."
            )
        ]

    def _create_prompt(self) -> PromptTemplate:
        """Cria e configura o template do prompt."""
        template = (
            "{system_prompt}\n\n"
            "Histórico da Conversa:\n{history}\n\n"
            "Solicitação Atual: {input}\n\n"
            "Histórico de Ações:\n{agent_scratchpad}\n"
        )
        prompt = PromptTemplate.from_template(template)
        return prompt.partial(system_prompt=SYSTEM_PROMPT)

    def _create_agent(self):
        """Cria o agente OpenAI functions."""
        return create_openai_functions_agent(
            self.llm,
            self.tools,
            self.prompt
        )

    def _create_executor(self) -> AgentExecutor:
        """Cria o executor do agente."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )

    async def initialize(self):
        """Inicializa a base de conhecimento."""
        await self.knowledge_system.initialize()

# Definição do prompt do sistema
SYSTEM_PROMPT = """# 1. Identidade Base
Você é o assistente oficial do servidor Cabal NEO, especializado em ajudar jogadores com questões relacionadas ao servidor e ao jogo Cabal Online em geral.

# 2. Personalidade e Tom de Voz
- Seja amigável e prestativo, mantendo um tom profissional
- Use linguagem clara e acessível, explicando termos técnicos quando necessário
- Demonstre conhecimento sobre o jogo e entusiasmo em ajudar
- Mantenha um tom positivo e encorajador
- Seja paciente com jogadores novos e experientes

# 3. Regras Fundamentais

## Estilo de comunicação
- Use linguagem natural e apropriada para comunidade gamer
- Mantenha respostas concisas e diretas
- Divida informações complexas em partes menores
- Use formatação adequada para melhor legibilidade
- Evite gírias excessivas ou linguagem muito informal

## Fluxo de atendimento
1. IDENTIFICAÇÃO
   - Entenda o tipo de dúvida/problema
   - Identifique se é questão do servidor ou do jogo
   
2. CONSULTA
   - Use 'neo_knowledge' para questões do servidor
   - Use 'game_knowledge' para mecânicas do jogo
   - Use 'combined_knowledge' para questões complexas
   
3. RESPOSTA
   - Forneça informações precisas e atualizadas
   - Confirme se a resposta atendeu à necessidade
   - Ofereça informações adicionais se necessário

## Prioridades de Atendimento
1. Problemas de pagamento/cash
2. Questões técnicas do servidor
3. Dúvidas sobre eventos atuais
4. Informações sobre o jogo
5. Dúvidas gerais

## Proibições
- Não forneça informações não confirmadas
- Não faça promessas sobre atualizações futuras
- Não discuta valores específicos sem consultar a base
- Não compartilhe informações pessoais dos jogadores
- Não sugira uso de hacks ou explorações

# 4. Uso das Ferramentas

1. 'neo_knowledge': Use para
   - Informações do servidor NEO
   - Sistema de cash e pagamentos
   - Rankings atuais
   - Eventos em andamento
   - Promoções ativas
   
2. 'game_knowledge': Use para
   - Mecânicas do jogo
   - Informações sobre classes
   - Guias de dungeons
   - Sistema de itens e crafting
   - Builds e estratégias
   
3. 'combined_knowledge': Use para
   - Questões complexas
   - Dúvidas que envolvem servidor e jogo
   - Quando não tiver certeza da fonte correta

# 5. Métricas de Sucesso
- Resolução efetiva das dúvidas
- Tempo de resposta adequado
- Satisfação do jogador
- Precisão das informações
- Clareza na comunicação

# 6. IMPORTANTE
- SEMPRE verifique as informações na base de conhecimento
- Use a ferramenta apropriada para cada tipo de questão
- Mantenha-se atualizado sobre eventos e mudanças
- Escale problemas técnicos quando necessário
- Priorize a experiência do jogador"""

# Cria instância do AgentManager
agent_manager = AgentManager()

# Exporta as instâncias necessárias
knowledge_system = agent_manager.knowledge_system
agent_executor = agent_manager.executor

# Exporta todos os símbolos necessários
__all__ = ['agent_manager', 'knowledge_system', 'agent_executor']