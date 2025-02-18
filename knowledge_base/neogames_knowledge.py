import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class KnowledgeSource(Enum):
    """Enumeração das diferentes seções do site (exceto rankings, que já são tratados separadamente)"""
    MAIN = "main"
    NEWS = "news"
    FAQ = "faq"
    DOWNLOAD = "download"
    SYSTEM = "system"
    SHOP = "shop"
    RECHARGE = "recharge"
    VIP = "vip"
    CUSTOM = "custom"

@dataclass
class SitemapEntry:
    """Estrutura para entrada do sitemap"""
    url: str
    lastmod: Optional[datetime] = None
    priority: float = 0.5

class NeoGamesKnowledge:
    def __init__(self, base_dir: str = "knowledge_base"):
        """
        Inicializa a base de conhecimento do NeoGames
        
        Args:
            base_dir: Diretório base para armazenar os dados
        """
        self.base_dir = base_dir
        self.sitemap_url = "https://www.neogames.online/sitemap.xml"
        self.base_url = "https://www.neogames.online"
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores: Dict[KnowledgeSource, Optional[FAISS]] = {
            source: None for source in KnowledgeSource
        }
        self._monitor_task = None
        
        # Criar diretórios para cada fonte
        self._create_directories()

    def _create_directories(self):
        """Cria a estrutura de diretórios necessária para cada fonte"""
        for source in KnowledgeSource:
            dir_path = os.path.join(self.base_dir, source.value)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Criado diretório para {source.value}: {dir_path}")

    def fetch_sitemap(self) -> Dict[KnowledgeSource, List[SitemapEntry]]:
        """
        Busca e processa o sitemap do site
        
        Returns:
            Dict com URLs organizadas por fonte
        """
        organized: Dict[KnowledgeSource, List[SitemapEntry]] = {
            source: [] for source in KnowledgeSource
        }
        
        try:
            response = requests.get(self.sitemap_url)
            response.raise_for_status()
            tree = ET.fromstring(response.content)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            for url_elem in tree.findall("ns:url", ns):
                try:
                    loc = url_elem.find("ns:loc", ns).text.strip()
                    lastmod_elem = url_elem.find("ns:lastmod", ns)
                    priority_elem = url_elem.find("ns:priority", ns)
                    
                    lastmod = None
                    if lastmod_elem is not None and lastmod_elem.text:
                        try:
                            lastmod = datetime.fromisoformat(lastmod_elem.text.replace('Z', '+00:00'))
                        except ValueError:
                            logger.warning(f"Formato de data inválido para {loc}")
                    
                    priority = 0.5
                    if priority_elem is not None and priority_elem.text:
                        try:
                            priority = float(priority_elem.text)
                        except ValueError:
                            logger.warning(f"Prioridade inválida para {loc}")
                    
                    entry = SitemapEntry(url=loc, lastmod=lastmod, priority=priority)
                    
                    parsed = urlparse(loc)
                    path = parsed.path.lower().strip("/")
                    
                    if not path:
                        organized[KnowledgeSource.MAIN].append(entry)
                    elif path.startswith("news"):
                        organized[KnowledgeSource.NEWS].append(entry)
                    elif path.startswith("faq"):
                        organized[KnowledgeSource.FAQ].append(entry)
                    elif path.startswith("download"):
                        organized[KnowledgeSource.DOWNLOAD].append(entry)
                    elif path.startswith("system"):
                        organized[KnowledgeSource.SYSTEM].append(entry)
                    elif path.startswith("vip"):
                        organized[KnowledgeSource.VIP].append(entry)
                    else:
                        organized[KnowledgeSource.CUSTOM].append(entry)
                    
                except Exception as e:
                    logger.error(f"Erro ao processar entrada do sitemap: {e}")
                    continue
            
            # Adicionar URLs extras que não estão no sitemap
            organized[KnowledgeSource.SHOP].append(
                SitemapEntry(url=f"{self.base_url}/shop")
            )
            organized[KnowledgeSource.RECHARGE].append(
                SitemapEntry(url=f"{self.base_url}/recharge")
            )
            organized[KnowledgeSource.VIP].append(
                SitemapEntry(url=f"{self.base_url}/vip")
            )
            
            return organized
            
        except Exception as e:
            logger.error(f"Erro ao buscar sitemap: {e}")
            return organized

    async def load_content(self, source: KnowledgeSource, urls: List[str]) -> List[Document]:
        """
        Carrega o conteúdo das URLs usando PlaywrightURLLoader
        
        Args:
            source: Fonte do conhecimento
            urls: Lista de URLs para carregar
            
        Returns:
            Lista de documentos processados
        """
        if not urls:
            return []
            
        try:
            loader = PlaywrightURLLoader(
                urls=urls,
                remove_selectors=[
                    "nav", "footer", "header", ".modal",
                    "script", "noscript", "style"
                ]
            )
            documents = await loader.aload()
            processed_docs = []
            
            for doc in documents:
                try:
                    soup = BeautifulSoup(doc.page_content, 'html.parser')
                    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    main_content = (
                        soup.find('main') or 
                        soup.find('div', {'role': 'main'}) or 
                        soup.find('div', class_=['content', 'main-content']) or
                        soup
                    )
                    
                    clean_content = main_content.get_text(separator=' ', strip=True)
                    
                    if clean_content:
                        processed_docs.append(Document(
                            page_content=clean_content,
                            metadata={
                                'source': source.value,
                                'url': doc.metadata.get('source'),
                                'timestamp': datetime.now().isoformat()
                            }
                        ))
                        
                except Exception as e:
                    logger.error(f"Erro ao processar documento: {e}")
                    continue
                    
            return processed_docs
            
        except Exception as e:
            logger.error(f"Erro ao carregar documentos: {e}")
            return []

    def create_knowledge_base(self, documents: List[Document], source: KnowledgeSource):
        """
        Cria ou atualiza a base de conhecimento para uma fonte
        
        Args:
            documents: Lista de documentos processados
            source: Fonte do conhecimento
        """
        try:
            if not documents:
                logger.warning(f"Nenhum documento para criar base {source.value}")
                return
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". "]
            )
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            save_path = os.path.join(self.base_dir, source.value)
            vectorstore.save_local(save_path)
            
            self.vectorstores[source] = vectorstore
            logger.info(f"Base {source.value} criada/atualizada com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao criar base {source.value}: {e}")

    async def initialize(self):
        """Inicializa as bases de conhecimento"""
        try:
            logger.info("Inicializando bases de conhecimento do NeoGames...")
            await self.update_knowledge_bases()
            self._monitor_task = asyncio.create_task(self._monitor_updates())
        except Exception as e:
            logger.error(f"Erro na inicialização das bases: {str(e)}")
            raise

    async def _monitor_updates(self):
        """Monitora atualizações periodicamente"""
        while True:
            try:
                await asyncio.sleep(3600 * 6)  # A cada 6 horas
                await self.update_knowledge_bases()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {str(e)}")
                await asyncio.sleep(60)

    async def shutdown(self):
        """Desliga corretamente o monitoramento"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def update_knowledge_bases(self):
        """Atualiza todas as bases de conhecimento"""
        try:
            sitemap_entries = self.fetch_sitemap()
            for source in KnowledgeSource:
                entries = sitemap_entries.get(source, [])
                if not entries:
                    continue
                    
                urls = [entry.url for entry in entries]
                documents = await self.load_content(source, urls)
                self.create_knowledge_base(documents, source)
                
        except Exception as e:
            logger.error(f"Erro ao atualizar bases: {e}")

    def query(self, question: str, sources: Optional[List[KnowledgeSource]] = None, k: int = 3) -> str:
        """
        Consulta as bases de conhecimento
        
        Args:
            question: Pergunta/consulta
            sources: Lista de fontes para consultar (opcional)
            k: Número de resultados por fonte
            
        Returns:
            Resposta formatada com os resultados
        """
        try:
            if not sources:
                sources = list(KnowledgeSource)
                
            all_docs = []
            
            for source in sources:
                if self.vectorstores[source]:
                    try:
                        docs = self.vectorstores[source].similarity_search(question, k=k)
                        all_docs.extend(docs)
                    except Exception as e:
                        logger.error(f"Erro na consulta à fonte {source.value}: {e}")
                        
            all_docs = sorted(all_docs, key=lambda x: x.metadata.get('score', 0), reverse=True)[:k]
            
            if not all_docs:
                return "Nenhuma informação relevante encontrada."
                
            responses = []
            for doc in all_docs:
                src = doc.metadata.get('source', 'desconhecida')
                url = doc.metadata.get('url', '')
                response = f"[{src.title()}]\n{doc.page_content}"
                if url:
                    response += f"\nFonte: {url}"
                responses.append(response)
                
            return "\n\n".join(responses)
            
        except Exception as e:
            logger.error(f"Erro na consulta: {e}")
            return "Erro ao consultar a base de conhecimento."
