import os
import logging
import re
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import torch

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


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
        
        # Define o diretório do vectorstore do site
        self.site_dir = os.path.join(self.base_dir, "neo_site")
        self.vectorstore_dir = os.path.join(self.site_dir, "faiss_store")
        
        # Único vectorstore para todo o site
        self.vectorstore: Optional[FAISS] = None
        
        # Mantém dicionário para URLs manuais (para organização)
        self.manual_urls: Dict[KnowledgeSource, List[str]] = {
            source: [] for source in KnowledgeSource
        }
        
        # Criar diretórios
        self._create_directories()
        
        # Inicializa o monitor task
        self._monitor_task = None
        
        # Inicializa URLs manuais conhecidas
        self._initialize_manual_urls()
        
        # Inicializa o embeddings com um modelo multilíngue
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            )
            logger.info("HuggingFaceEmbeddings inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar HuggingFaceEmbeddings: {e}")
            raise

    def _create_directories(self):
        """Cria a estrutura de diretórios necessária"""
        try:
            # Cria diretório para o site
            os.makedirs(self.site_dir, exist_ok=True)
            os.makedirs(self.vectorstore_dir, exist_ok=True)
            logger.info(f"Criados diretórios: {self.site_dir}, {self.vectorstore_dir}")
        except Exception as e:
            logger.error(f"Erro ao criar diretórios: {e}")
            raise

    def _initialize_manual_urls(self):
        """Inicializa URLs conhecidas que podem não estar no sitemap"""
        base = self.base_url.rstrip('/')
        
        # URLs de notícias importantes
        self.manual_urls[KnowledgeSource.NEWS].extend([
            f"{base}/news/como-obter-asa-arcana-colecao-e-link-estelar",
            f"{base}/news/aniversario-neo-2025",
            f"{base}/news/halloween-neo-2024",
            f"{base}/news/natal-neo-2024",
            f"{base}/news/primavera-neo-2024"
        ])
        
        # URLs do sistema
        self.manual_urls[KnowledgeSource.SYSTEM].extend([
            f"{base}/system/debuffs-personalizados",
            f"{base}/system/drop-anuncio",
            f"{base}/system/informacao-de-sinergia",
            f"{base}/system/macro-de-bm3",
            f"{base}/system/modo-de-batalha",
            f"{base}/system/quebra-de-itens-automatica",
            f"{base}/system/seguro-neo"
        ])
        
        # Outras URLs importantes
        self.manual_urls[KnowledgeSource.SHOP].extend([
            f"{base}/shop",
            f"{base}/shop/items"
        ])
        self.manual_urls[KnowledgeSource.RECHARGE].extend([
            f"{base}/recharge",
            f"{base}/recharge/methods"
        ])
        self.manual_urls[KnowledgeSource.VIP].extend([
            f"{base}/vip",
            f"{base}/vip/benefits"
        ])

    def add_manual_url(self, source: KnowledgeSource, url: str) -> bool:
        """
        Adiciona uma URL manualmente à base de conhecimento
        
        Args:
            source: Categoria da URL
            url: URL completa ou path
            
        Returns:
            bool: True se adicionada com sucesso
        """
        try:
            # Normaliza a URL
            if not url.startswith(('http://', 'https://')):
                url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
            
            # Verifica se já existe
            if url in self.manual_urls[source]:
                logger.info(f"URL já existe em {source.value}: {url}")
                return False
                
            self.manual_urls[source].append(url)
            logger.info(f"URL adicionada com sucesso a {source.value}: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar URL manual: {e}")
            return False

    def remove_manual_url(self, source: KnowledgeSource, url: str) -> bool:
        """
        Remove uma URL manual da base de conhecimento
        
        Args:
            source: Categoria da URL
            url: URL a ser removida
            
        Returns:
            bool: True se removida com sucesso
        """
        try:
            # Normaliza a URL
            if not url.startswith(('http://', 'https://')):
                url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
                
            if url in self.manual_urls[source]:
                self.manual_urls[source].remove(url)
                logger.info(f"URL removida de {source.value}: {url}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Erro ao remover URL manual: {e}")
            return False

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
            
            # Adiciona URLs manuais
            for source, urls in self.manual_urls.items():
                for url in urls:
                    if not any(entry.url == url for entry in organized[source]):
                        organized[source].append(
                            SitemapEntry(
                                url=url,
                                lastmod=datetime.now(UTC),
                                priority=0.8  # Prioridade alta para URLs manuais
                            )
                        )
            
            return organized
            
        except Exception as e:
            logger.error(f"Erro ao buscar sitemap: {e}")
            return organized

    async def load_content(self, source: KnowledgeSource, urls: List[str]) -> List[Document]:
        if not urls:
            return []
                
        try:
            absolute_urls = [
                urljoin(self.base_url, url) if not url.startswith(('http://', 'https://')) else url
                for url in urls
            ]
            
            loader = PlaywrightURLLoader(
                urls=absolute_urls,
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
                    post_date = None
                    
                    # 1. Tenta encontrar a data no conteúdo da notícia
                    date_patterns = [
                        r'(\d{2}/\d{2}/\d{4})',  # dd/mm/yyyy
                        r'(\d{2}-\d{2}-\d{4})',  # dd-mm-yyyy
                        r'(\d{2}/\d{2}/\d{2})',  # dd/mm/yy
                        r'(\d{2}-\d{2}-\d{2})',  # dd-mm-yy
                    ]
                    
                    # Procura no texto completo primeiro
                    text_content = soup.get_text()
                    for pattern in date_patterns:
                        match = re.search(pattern, text_content)
                        if match:
                            date_str = match.group(1)
                            try:
                                if '-' in date_str:
                                    if len(date_str) == 8:  # dd-mm-yy
                                        post_date = datetime.strptime(date_str, '%d-%m-%y')
                                    else:  # dd-mm-yyyy
                                        post_date = datetime.strptime(date_str, '%d-%m-%Y')
                                else:  # formato com /
                                    if len(date_str) == 8:  # dd/mm/yy
                                        post_date = datetime.strptime(date_str, '%d/%m/%y')
                                    else:  # dd/mm/yyyy
                                        post_date = datetime.strptime(date_str, '%d/%m/%Y')
                                break
                            except ValueError:
                                continue
                    
                    # 2. Se não encontrou, tenta extrair da URL
                    if not post_date:
                        url = doc.metadata.get('source', '')
                        date_matches = re.findall(r'(\d{2}-\d{2}-\d{2}(?:\d{2})?)', url)
                        for date_str in date_matches:
                            try:
                                if len(date_str) == 8:  # dd-mm-yy
                                    post_date = datetime.strptime(date_str, '%d-%m-%y')
                                else:  # dd-mm-yyyy
                                    post_date = datetime.strptime(date_str, '%d-%m-%Y')
                                break
                            except ValueError:
                                continue
                    
                    # 3. Se ainda não encontrou, usa a data atual como último recurso
                    if not post_date:
                        post_date = datetime.now(UTC)
                    
                    # Limpa o conteúdo
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
                        metadata = {
                            'source': source.value,
                            'url': doc.metadata.get('source'),
                            'timestamp': datetime.now(UTC).isoformat(),
                            'post_date': post_date.astimezone(UTC).isoformat() if post_date else None
                        }
                                                
                        # Remove o "Voltar para notícias" do conteúdo
                        clean_content = clean_content.replace('Voltar para notícias', '').strip()
                        
                        processed_docs.append(Document(
                            page_content=clean_content,
                            metadata=metadata
                        ))
                        
                except Exception as e:
                    logger.error(f"Erro ao processar documento: {e}")
                    continue
                        
            return processed_docs
                
        except Exception as e:
            logger.error(f"Erro ao carregar documentos: {e}")
            return []

    def create_knowledge_base(self, documents: List[Document]):
        """
        Cria ou atualiza a base de conhecimento unificada
        
        Args:
            documents: Lista de documentos processados
        """
        try:
            if not documents:
                logger.warning("Nenhum documento para criar base")
                return
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". "]
            )
            splits = text_splitter.split_documents(documents)
            
            # Cria ou atualiza o vectorstore único
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            else:
                self.vectorstore.add_documents(splits)
                
            # Salva no diretório específico do site
            self.vectorstore.save_local(self.vectorstore_dir)
            logger.info("Base de conhecimento atualizada com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao criar base: {e}")

    async def initialize(self):
        """Inicializa a base de conhecimento"""
        try:
            logger.info("Inicializando base de conhecimento do NeoGames...")
            
            # Tenta carregar base existente
            if os.path.exists(self.vectorstore_dir):
                try:
                    self.vectorstore = FAISS.load_local(
                        self.vectorstore_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True  # Adicionado este parâmetro

                    )
                    logger.info("Base de conhecimento existente carregada")
                except Exception as e:
                    logger.warning(f"Erro ao carregar base existente: {e}")
                    self.vectorstore = None
            
            # Atualiza a base
            await self.update_knowledge_bases()
            
            # Inicia monitoramento
            self._monitor_task = asyncio.create_task(self._monitor_updates())
            
        except Exception as e:
            logger.error(f"Erro na inicialização da base: {str(e)}")
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
        """Atualiza a base de conhecimento unificada"""
        try:
            sitemap_entries = self.fetch_sitemap()
            all_documents = []
            
            # Processa URLs regulares e manuais
            for source in KnowledgeSource:
                entries = sitemap_entries.get(source, [])
                if not entries:
                    continue
                    
                urls = [entry.url for entry in entries]
                
                # Adiciona URLs manuais se existirem
                manual_urls = self.manual_urls.get(source, [])
                if manual_urls:
                    urls.extend(manual_urls)
                
                # Remove duplicatas mantendo a ordem
                urls = list(dict.fromkeys(urls))
                
                # Remove o terceiro parâmetro que estava causando o erro
                documents = await self.load_content(source, urls)
                all_documents.extend(documents)
            
            # Cria/atualiza a base unificada
            self.create_knowledge_base(all_documents)
            
        except Exception as e:
            logger.error(f"Erro ao atualizar bases: {e}")

    def query(self, question: str, sources: Optional[List[KnowledgeSource]] = None, k: int = 3) -> str:
        try:
            if not self.vectorstore:
                return "Base de conhecimento não inicializada."

            # Busca em toda a base
            docs = self.vectorstore.similarity_search(question, k=k*2)
            
            # Filtra por fonte se necessário
            if sources:
                docs = [
                    doc for doc in docs 
                    if doc.metadata.get('source') in [s.value for s in sources]
                ]
            
            # Função auxiliar para obter data com timezone
            def get_date(doc) -> datetime:
                try:
                    post_date = doc.metadata.get('post_date')
                    if isinstance(post_date, str):
                        # Converte string para datetime e garante timezone UTC
                        dt = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
                        return dt.astimezone(UTC)
                    elif isinstance(post_date, datetime):
                        # Se já é datetime, garante timezone UTC
                        return post_date.astimezone(UTC) if post_date.tzinfo else post_date.replace(tzinfo=UTC)
                    else:
                        # Tenta timestamp como fallback
                        timestamp = doc.metadata.get('timestamp')
                        if timestamp and isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            return dt.astimezone(UTC)
                        return datetime.min.replace(tzinfo=UTC)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Erro ao processar data: {e}")
                    return datetime.min.replace(tzinfo=UTC)
            
            # Ordena documentos por data
            try:
                docs = sorted(
                    docs,
                    key=get_date,
                    reverse=True
                )[:k]
            except Exception as e:
                logger.error(f"Erro ao ordenar documentos: {e}")
                # Se falhar na ordenação, usa os documentos sem ordenar
                docs = docs[:k]
            
            if not docs:
                return "Nenhuma informação relevante encontrada."
            
            responses = []
            for doc in docs:
                try:
                    src = doc.metadata.get('source', 'desconhecida').title()
                    url = doc.metadata.get('url', '')
                    
                    # Formata a data
                    try:
                        post_date = get_date(doc)
                        date_str = post_date.strftime('%d/%m/%Y %H:%M')
                    except Exception as e:
                        logger.debug(f"Erro ao formatar data: {e}")
                        date_str = "Data não disponível"
                    
                    # Remove texto desnecessário e formata o conteúdo
                    content = doc.page_content.strip()
                    if not content:
                        continue
                        
                    # Remove textos específicos indesejados
                    content = content.replace('Voltar para notícias', '')
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    response = f"[{src}] [{date_str}]\n{content}"
                    if url:
                        response += f"\nFonte: {url}"
                    responses.append(response)
                    
                except Exception as e:
                    logger.error(f"Erro ao processar documento individual: {e}")
                    continue
            
            return "\n\n".join(responses) if responses else "Nenhuma informação relevante encontrada."
            
        except Exception as e:
            logger.error(f"Erro na consulta: {e}")
            return "Erro ao consultar a base de conhecimento."

    def get_all_urls(self) -> Dict[KnowledgeSource, List[str]]:
        """
        Retorna todas as URLs conhecidas, incluindo manuais e do sitemap
        
        Returns:
            Dict com todas as URLs organizadas por fonte
        """
        try:
            all_urls = {source: [] for source in KnowledgeSource}
            sitemap_entries = self.fetch_sitemap()
            
            for source in KnowledgeSource:
                # URLs do sitemap
                urls = [entry.url for entry in sitemap_entries.get(source, [])]
                
                # Adiciona URLs manuais
                manual_urls = self.manual_urls.get(source, [])
                urls.extend(manual_urls)
                
                # Remove duplicatas mantendo a ordem
                all_urls[source] = list(dict.fromkeys(urls))
            
            return all_urls
            
        except Exception as e:
            logger.error(f"Erro ao obter todas as URLs: {e}")
            return {source: [] for source in KnowledgeSource}