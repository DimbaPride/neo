# knowledge_base/knowledge_system.py
import os
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PlaywrightURLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class KnowledgeSource(Enum):
    """Fontes de conhecimento do sistema"""
    NEO_CABAL = auto()  # Site oficial do Cabal NEO (fonte primária)
    MRWORMY = auto()    # Base de conhecimento do jogo (fonte secundária)

class NeoSection(Enum):
    """Seções do site Neo Cabal"""
    HOME = ""
    NEWS = "news"
    DOWNLOAD = "download"
    FAQ = "faq"
    VIP = "vip"
    RECHARGE = "recharge"
    SHOP = "shop"
    RANKING_WAR = "ranking/war"
    RANKING_POWER = "ranking/power"
    RANKING_GUILD = "ranking/guild"
    RANKING_MEMORIAL = "ranking/memorial"

class MrWormySection(Enum):
    """Seções importantes do MrWormy para conhecimento do jogo"""
    HOME = ""
    CLASSES = "classes"
    DUNGEONS = "dungeons"
    ITEMS = "items"
    SKILLS = "skills"
    SYSTEMS = "systems"
    MECHANICS = "mechanics"

@dataclass
class SourceConfig:
    """Configuração para fonte de conhecimento"""
    source_type: KnowledgeSource
    urls: List[str]
    update_interval: int = 86400  # 24 horas
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContentProcessor(ABC):
    """Processador base de conteúdo"""
    @abstractmethod
    async def process(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        pass

class NeoProcessor(ContentProcessor):
    """Processador específico para o site do Neo Cabal"""
    async def process(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        soup = BeautifulSoup(content, 'html.parser')
        url = metadata.get('url', '')
        
        # Remove elementos irrelevantes
        for element in soup.select('script, style, nav, footer, header, .ads'):
            element.decompose()
        
        section = self._get_section(url)
        
        if 'ranking' in url:
            return self._process_ranking(soup, metadata)
        elif 'news' in url:
            return self._process_news(soup, metadata)
        elif 'shop' in url:
            return self._process_shop(soup, metadata)
        elif 'faq' in url:
            return self._process_faq(soup, metadata)
        else:
            return self._process_general(soup, metadata)
    
    def _get_section(self, url: str) -> str:
        for section in NeoSection:
            if section.value in url:
                return section.value
        return "general"
    
    def _process_ranking(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        tables = soup.find_all('table')
        documents = []
        
        for table in tables:
            rows = table.find_all('tr')
            content = []
            for row in rows:
                cells = row.find_all(['th', 'td'])
                content.append([cell.get_text(strip=True) for cell in cells])
                
            if content:
                documents.append(Document(
                    page_content=str(content),
                    metadata={**metadata, 'type': 'ranking'}
                ))
        
        return documents
    
    def _process_news(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        news_items = soup.find_all(['article', 'div.news-item'])
        documents = []
        
        for item in news_items:
            title = item.find(['h1', 'h2', 'h3'])
            content = item.find(['div.content', 'div.description'])
            date = item.find(['time', 'div.date'])
            
            if title and content:
                doc_content = f"{title.get_text()}\n\n{content.get_text()}"
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        **metadata,
                        'type': 'news',
                        'date': date.get_text() if date else None
                    }
                ))
        
        return documents
    
    def _process_faq(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        faqs = soup.find_all(['div.faq-item', 'div.qa'])
        documents = []
        
        for faq in faqs:
            question = faq.find(['h3', 'div.question'])
            answer = faq.find(['div.answer', 'div.content'])
            
            if question and answer:
                content = f"Q: {question.get_text()}\nA: {answer.get_text()}"
                documents.append(Document(
                    page_content=content,
                    metadata={**metadata, 'type': 'faq'}
                ))
        
        return documents
    
    def _process_shop(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        items = soup.find_all(['div.shop-item', 'div.item'])
        documents = []
        
        for item in items:
            name = item.find(['h3', 'div.name'])
            desc = item.find(['div.description', 'div.content'])
            price = item.find(['div.price', 'span.price'])
            
            if name:
                content = [f"Item: {name.get_text()}"]
                if desc:
                    content.append(f"Description: {desc.get_text()}")
                if price:
                    content.append(f"Price: {price.get_text()}")
                    
                documents.append(Document(
                    page_content='\n'.join(content),
                    metadata={**metadata, 'type': 'shop_item'}
                ))
        
        return documents
    
    def _process_general(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        content = main.get_text(strip=True, separator='\n') if main else soup.get_text(strip=True, separator='\n')
        
        return [Document(
            page_content=content,
            metadata={**metadata, 'type': 'general'}
        )]

class MrWormyProcessor(ContentProcessor):
    """Processador específico para o site MrWormy (conhecimento do jogo)"""
    async def process(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        soup = BeautifulSoup(content, 'html.parser')
        url = metadata.get('url', '')
        
        # Remove elementos irrelevantes
        for element in soup.select('script, style, nav, footer, header'):
            element.decompose()
            
        section = self._get_section(url)
        
        if 'classes' in url:
            return self._process_classes(soup, metadata)
        elif 'dungeons' in url:
            return self._process_dungeons(soup, metadata)
        elif 'items' in url:
            return self._process_items(soup, metadata)
        elif 'skills' in url:
            return self._process_skills(soup, metadata)
        elif 'systems' in url:
            return self._process_systems(soup, metadata)
        else:
            return self._process_general(soup, metadata)
    
    def _get_section(self, url: str) -> str:
        for section in MrWormySection:
            if section.value in url:
                return section.value
        return "general"
    
    def _process_classes(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        class_sections = soup.find_all(['div.class', 'article.class'])
        documents = []
        
        for section in class_sections:
            name = section.find(['h2', 'h3'])
            desc = section.find(['div.description', 'div.content'])
            skills = section.find_all(['div.skill', 'div.ability'])
            
            if name:
                content = [f"Class: {name.get_text()}"]
                if desc:
                    content.append(f"Description: {desc.get_text()}")
                if skills:
                    content.append("Skills:")
                    for skill in skills:
                        skill_name = skill.find(['h4', 'div.name'])
                        skill_desc = skill.find(['div.description', 'div.effect'])
                        if skill_name and skill_desc:
                            content.append(f"- {skill_name.get_text()}: {skill_desc.get_text()}")
                            
                documents.append(Document(
                    page_content='\n'.join(content),
                    metadata={**metadata, 'type': 'class_info'}
                ))
        
        return documents
    
    def _process_dungeons(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        dungeon_sections = soup.find_all(['div.dungeon', 'article.dungeon'])
        documents = []
        
        for section in dungeon_sections:
            name = section.find(['h2', 'h3'])
            desc = section.find(['div.description', 'div.content'])
            reqs = section.find(['div.requirements', 'div.prereq'])
            rewards = section.find(['div.rewards', 'div.drops'])
            
            if name:
                content = [f"Dungeon: {name.get_text()}"]
                if desc:
                    content.append(f"Description: {desc.get_text()}")
                if reqs:
                    content.append(f"Requirements: {reqs.get_text()}")
                if rewards:
                    content.append(f"Rewards: {rewards.get_text()}")
                    
                documents.append(Document(
                    page_content='\n'.join(content),
                    metadata={**metadata, 'type': 'dungeon_info'}
                ))
        
        return documents
    
    def _process_general(self, soup: BeautifulSoup, metadata: Dict[str, Any]) -> List[Document]:
        main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        content = main.get_text(strip=True, separator='\n') if main else soup.get_text(strip=True, separator='\n')
        
        return [Document(
            page_content=content,
            metadata={**metadata, 'type': 'general'}
        )]

class CabalKnowledgeSystem:
    """Sistema de conhecimento do Cabal"""
    def __init__(self):
        # Define o caminho base relativo ao diretório knowledge_base
        self.base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cabal_knowledge_base"
        )
        self.sources = {}
        self.processors = {
            KnowledgeSource.NEO_CABAL: NeoProcessor(),
            KnowledgeSource.MRWORMY: MrWormyProcessor()
        }
        self.vectorstores = {}
        self._configure_sources()
        
    def _configure_sources(self):
        """Configura as fontes de conhecimento"""
        # Configura Neo Cabal
        neo_base_url = "https://www.neogames.online"
        neo_urls = [f"{neo_base_url}/{section.value}" for section in NeoSection]
        neo_urls.append(neo_base_url)
        
        self.sources[KnowledgeSource.NEO_CABAL] = SourceConfig(
            source_type=KnowledgeSource.NEO_CABAL,
            urls=neo_urls,
            metadata={'priority': 1, 'source': 'neo_cabal'}
        )
        
        # Configura MrWormy
        mrwormy_base_url = "https://mrwormy.com"
        mrwormy_urls = [f"{mrwormy_base_url}/{section.value}" for section in MrWormySection]
        mrwormy_urls.append(mrwormy_base_url)
        
        self.sources[KnowledgeSource.MRWORMY] = SourceConfig(
            source_type=KnowledgeSource.MRWORMY,
            urls=mrwormy_urls,
            metadata={'priority': 2, 'source': 'mrwormy'}
        )
    
    async def initialize(self):
        """Inicializa o sistema de conhecimento"""
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Cria diretórios para backups
        for source_type in KnowledgeSource:
            source_dir = os.path.join(self.base_dir, source_type.name.lower())
            backup_dir = os.path.join(source_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            try:
                if self._needs_update(source_type):
                    await self._update_source(source_type)
                else:
                    await self._load_source(source_type)
            except Exception as e:
                logger.error(f"Erro ao inicializar fonte {source_type.name}: {str(e)}")
                # Continua com a próxima fonte em caso de erro
                continue
    
    def _needs_update(self, source_type: KnowledgeSource) -> bool:
        source_dir = os.path.join(self.base_dir, source_type.name.lower())
        index_path = os.path.join(source_dir, "index.faiss")
        
        if not os.path.exists(index_path):
            return True
            
        last_update = os.path.getmtime(index_path)
        return (time.time() - last_update) > self.sources[source_type].update_interval
    
    async def _update_source(self, source_type: KnowledgeSource):
        """Atualiza uma fonte de conhecimento"""
        try:
            config = self.sources[source_type]
            processor = self.processors[source_type]
            documents = []

            # Se for MRWORMY, vai direto para os backups
            if source_type == KnowledgeSource.MRWORMY:
                documents = await self._load_backup_documents(source_type)
            else:
                # Tenta carregar da web apenas para NEO_CABAL
                loader = PlaywrightURLLoader(
                    urls=config.urls,
                    remove_selectors=['script', 'style', 'nav', 'footer', 'header', '.ads']
                )
                
                try:
                    raw_documents = await loader.aload()
                    # Processa documentos
                    for doc in raw_documents:
                        processed_docs = await processor.process(
                            doc.page_content,
                            {**config.metadata, 'url': doc.metadata.get('source')}
                        )
                        documents.extend(processed_docs)
                except Exception as e:
                    logger.warning(f"Erro ao carregar URLs para {source_type.name}: {str(e)}")
                    if source_type == KnowledgeSource.NEO_CABAL:
                        # Para NEO_CABAL, recarrega da fonte anterior se disponível
                        try:
                            return await self._load_source(source_type)
                        except:
                            raise  # Se não conseguir carregar, propaga o erro
            
            if not documents:
                documents = await self._load_backup_documents(source_type)
            
            if not documents:
                documents = self._create_default_documents(source_type)
            
            if documents:
                # Cria embeddings
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                chunks = splitter.split_documents(documents)
                
                # Cria e salva vectorstore
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                source_dir = os.path.join(self.base_dir, source_type.name.lower())
                vectorstore.save_local(source_dir)
                
                self.vectorstores[source_type] = vectorstore
                logger.info(f"Fonte {source_type.name} atualizada com sucesso!")
            else:
                logger.warning(f"Nenhum documento disponível para {source_type.name}")
                
        except Exception as e:
            logger.error(f"Erro ao atualizar fonte {source_type.name}: {str(e)}")
            raise

    async def _load_backup_documents(self, source_type: KnowledgeSource) -> List[Document]:
        """Carrega documentos de backup quando as URLs falham"""
        backup_dir = os.path.join(self.base_dir, source_type.name.lower(), "backup")
        documents = []

        if source_type == KnowledgeSource.MRWORMY:
            # Estrutura de backup para o MrWormy
            backup_files = {
                "classes.txt": "classes",
                "dungeons.txt": "dungeons",
                "items.txt": "items",
                "skills.txt": "skills",
                "systems.txt": "systems",
                "mechanics.txt": "mechanics"
            }

            for filename, doc_type in backup_files.items():
                file_path = os.path.join(backup_dir, filename)
                if os.path.exists(file_path):
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['type'] = doc_type
                            doc.metadata['source'] = 'backup'
                        documents.extend(docs)
                    except Exception as e:
                        logger.error(f"Erro ao carregar backup {filename}: {str(e)}")

        return documents or self._create_default_documents(source_type)
    
    def _create_default_documents(self, source_type: KnowledgeSource) -> List[Document]:
        """Cria documentos padrão quando não há backups disponíveis"""
        if source_type == KnowledgeSource.MRWORMY:
            # Conhecimento básico sobre o Cabal
            return [
                Document(
                    page_content="Cabal Online é um MMORPG que possui várias classes incluindo Force Blader, Force Shielder, Wizard, Force Archer, Gladiator, e outras.",
                    metadata={'type': 'classes', 'source': 'default'}
                ),
                Document(
                    page_content="O jogo possui diversos sistemas como crafting, enhancement, dungeons, PvP, e Guerra entre Nações.",
                    metadata={'type': 'systems', 'source': 'default'}
                )
            ]
        return []
    
    async def _load_source(self, source_type: KnowledgeSource):
        """Carrega uma fonte de conhecimento do disco"""
        try:
            source_dir = os.path.join(self.base_dir, source_type.name.lower())
            embeddings = OpenAIEmbeddings()
            
            self.vectorstores[source_type] = FAISS.load_local(
                source_dir, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Fonte {source_type.name} carregada com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar fonte {source_type.name}: {str(e)}")
            await self._update_source(source_type)
    
    async def query(self, question: str, sources: Optional[List[KnowledgeSource]] = None, k: int = 3) -> str:
        """Consulta as bases de conhecimento"""
        query_sources = sources or list(self.sources.keys())
        results = []
        
        for source_type in query_sources:
            if source_type not in self.vectorstores:
                continue
                
            try:
                # Fazendo a busca de forma assíncrona
                docs = await self.vectorstores[source_type].asimilarity_search(question, k=k)
                for doc in docs:
                    doc.metadata['source_type'] = source_type.name
                results.extend(docs)
            except Exception as e:
                logger.error(f"Erro ao consultar {source_type.name}: {str(e)}")
        
        return self._format_response(results[:k])
    
    def _format_response(self, docs: List[Document]) -> str:
        """Formata a resposta com base nos documentos encontrados"""
        if not docs:
            return "Desculpe, não encontrei informações relevantes para sua pergunta."
        
        formatted_responses = []
        for doc in docs:
            source = doc.metadata.get('source_type', 'Desconhecido')
            doc_type = doc.metadata.get('type', 'general')
            
            header = f"[{source} - {doc_type}]"
            formatted_responses.append(f"{header}\n{doc.page_content}")
        
        return "\n\n".join(formatted_responses)

# Exporta apenas as classes e funções necessárias
__all__ = ['CabalKnowledgeSystem', 'KnowledgeSource']