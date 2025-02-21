#knowledge_base/neogames_knowledge.py
import os
import logging
import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
import torch

import requests
from rank_bm25 import BM25Okapi
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
        self.bm25_index = None
        self.documents_for_bm25 = []
        
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
                model_name="intfloat/multilingual-e5-large",
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
        ])
        
        # URLs do sistema
        self.manual_urls[KnowledgeSource.SYSTEM].extend([
        ])
        
        # Outras URLs importantes
        self.manual_urls[KnowledgeSource.SHOP].extend([
            f"{base}/shop",            
        ])
        self.manual_urls[KnowledgeSource.RECHARGE].extend([
            f"{base}/recharge",            
        ])
        self.manual_urls[KnowledgeSource.VIP].extend([
            f"{base}/vip",            
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
                    elif path.startswith("news"):  # Removido /
                        organized[KnowledgeSource.NEWS].append(entry)
                    elif path.startswith("faq"):   # Removido /
                        organized[KnowledgeSource.FAQ].append(entry)
                    elif path.startswith("download"):  # Removido /
                        organized[KnowledgeSource.DOWNLOAD].append(entry)
                    elif path.startswith("system"):   # Removido /
                        organized[KnowledgeSource.SYSTEM].append(entry)
                    elif path.startswith("vip"):     # Removido /
                        organized[KnowledgeSource.VIP].append(entry)
                    elif path.startswith("shop"):    # Removido /
                        organized[KnowledgeSource.SHOP].append(entry)
                    elif path.startswith("recharge"): # Removido /
                        organized[KnowledgeSource.RECHARGE].append(entry)
                    
                except Exception as e:
                    logger.error(f"Erro ao processar entrada do sitemap: {e}")
                    continue
            
            # Adiciona URLs manuais
            for source, urls in self.manual_urls.items():
                logger.info(f"Processando URLs manuais para {source.value}")
                for url in urls:
                    if not any(entry.url == url for entry in organized[source]):
                        logger.info(f"Adicionando URL manual: {url}")
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
            # Converte URLs relativas para absolutas
            logger.info(f"Tentando carregar {len(urls)} URLs para {source.value}")
            absolute_urls = []
            for url in urls:
                if not url.startswith(('http://', 'https://')):
                    absolute_url = urljoin(self.base_url, url)
                    absolute_urls.append(absolute_url)
                else:
                    absolute_urls.append(url)
            
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

    def _prepare_bm25_index(self):
        """Prepara o índice BM25 com os documentos atuais"""
        try:
            if not self.vectorstore:
                return
                    
            # Nova forma de acessar os documentos
            documents = []
            docstore = self.vectorstore.docstore
            # Pegar todos os documentos do docstore
            for doc_id in docstore._dict:  # Usando _dict ao invés de documents
                doc = docstore._dict[doc_id]
                if hasattr(doc, 'page_content'):
                    documents.append(doc)
            
            logging.info(f"Preparando BM25 com {len(documents)} documentos")
            
            # Preparar corpus para BM25
            tokenized_corpus = []
            self.documents_for_bm25 = []
            
            for doc in documents:
                # Tokenizar o texto em palavras
                tokens = doc.page_content.lower().split()
                tokenized_corpus.append(tokens)
                self.documents_for_bm25.append(doc)
            
            # Criar índice BM25
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logging.info("Índice BM25 preparado com sucesso")
                
        except Exception as e:
            logging.error(f"Erro ao preparar índice BM25: {str(e)}", exc_info=True)
            self.bm25_index = None

    def create_knowledge_base(self, documents: List[Document]):
        try:
            if not documents:
                logging.warning("Nenhum documento para criar base")
                return
                    
            logging.info(f"Iniciando split de {len(documents)} documentos")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". "]
            )
            splits = text_splitter.split_documents(documents)
            logging.info(f"Split concluído: {len(splits)} chunks")
                
            # Criar ou atualizar o vectorstore
            logging.info("Iniciando criação/atualização do vectorstore")
            
            try:
                # Teste de memória antes de criar
                logging.info("Verificando memoria disponível...")
                import psutil
                mem = psutil.virtual_memory()
                logging.info(f"Memória disponível: {mem.available / (1024 * 1024):.2f} MB")
                
                # Tentar criar em partes menores
                batch_size = 30
                all_splits = splits
                if self.vectorstore is None:
                    # Criar com primeiro batch
                    first_batch = all_splits[:batch_size]
                    logging.info(f"Criando vectorstore inicial com {len(first_batch)} documentos")
                    self.vectorstore = FAISS.from_documents(first_batch, self.embeddings)
                    
                    # Adicionar resto em batches
                    remaining = all_splits[batch_size:]
                    for i in range(0, len(remaining), batch_size):
                        batch = remaining[i:i + batch_size]
                        logging.info(f"Adicionando batch de {len(batch)} documentos")
                        self.vectorstore.add_documents(batch)
                else:
                    # Adicionar em batches
                    for i in range(0, len(all_splits), batch_size):
                        batch = all_splits[i:i + batch_size]
                        logging.info(f"Adicionando batch de {len(batch)} documentos")
                        self.vectorstore.add_documents(batch)
                        
                logging.info("Vectorstore atualizado com sucesso")
                    
            except Exception as e:
                logging.error(f"Erro na criação do vectorstore: {str(e)}", exc_info=True)
                raise
                    
            # Salvar vectorstore
            logging.info("Salvando vectorstore")
            self.vectorstore.save_local(self.vectorstore_dir)
                
            # Preparar índice BM25
            logging.info("Preparando índice BM25")
            self._prepare_bm25_index()
                
            logging.info("Base de conhecimento atualizada com sucesso")
                
        except Exception as e:
            logging.error(f"Erro ao criar base: {str(e)}", exc_info=True, stack_info=True)

    def hybrid_search(self, question: str, k: int = 3, sources: Optional[List[KnowledgeSource]] = None):
        """Realiza busca híbrida combinando BM25 e embeddings"""
        try:
            if not self.vectorstore or not self.bm25_index:
                return []
            
            # Filtrar por fonte antes da busca
            if sources:
                source_values = [s.value for s in sources]
                
                # Busca semântica com FAISS
                semantic_results = self.vectorstore.similarity_search(
                    question, 
                    k=k*2
                )
                
                # Filtrar resultados FAISS por fonte
                filtered_semantic = [
                    doc for doc in semantic_results 
                    if isinstance(doc, Document) and doc.metadata.get('source') in source_values
                ]
                
                # Busca e filtro BM25
                tokenized_query = question.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                
                # Filtrar resultados BM25
                bm25_results = []
                for idx in bm25_scores.argsort()[::-1]:
                    doc = self.documents_for_bm25[idx]
                    if isinstance(doc, Document) and doc.metadata.get('source') in source_values:
                        bm25_results.append(doc)
                    if len(bm25_results) >= k*2:
                        break
            else:
                # Busca sem filtro
                semantic_results = self.vectorstore.similarity_search(question, k=k*2)
                filtered_semantic = semantic_results
                
                tokenized_query = question.lower().split()
                bm25_scores = self.bm25_index.get_scores(tokenized_query)
                top_indices = bm25_scores.argsort()[-k*2:][::-1]
                bm25_results = [self.documents_for_bm25[i] for i in top_indices]
            
            # Combinar resultados
            combined_results = []
            seen_content = set()
            
            # Função auxiliar para adicionar documento
            def add_unique_doc(doc):
                if isinstance(doc, Document) and doc.page_content not in seen_content:
                    combined_results.append(doc)
                    seen_content.add(doc.page_content)
            
            # Adicionar resultados BM25 primeiro
            for doc in bm25_results:
                add_unique_doc(doc)
            
            # Depois adicionar resultados semânticos
            for doc in filtered_semantic:
                add_unique_doc(doc)
            
            logging.info(f"Resultados combinados: {len(combined_results)} documentos")
            return combined_results[:k]
                
        except Exception as e:
            logging.error(f"Erro na busca híbrida: {str(e)}", exc_info=True)
            return []

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
                        allow_dangerous_deserialization=True
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
        try:
            sitemap_entries = self.fetch_sitemap()
            all_documents = []
            
            # Adicionar logs para debug
            logger.info("Iniciando atualização das bases")
            
            for source in KnowledgeSource:
                entries = sitemap_entries.get(source, [])
                logger.info(f"Source {source.value}: {len(entries)} entradas do sitemap")
                
                if not entries:
                    logger.warning(f"Nenhuma entrada para {source.value}")
                    continue
                    
                urls = [entry.url for entry in entries]
                
                # Adiciona URLs manuais
                manual_urls = self.manual_urls.get(source, [])
                if manual_urls:
                    logger.info(f"Adicionando {len(manual_urls)} URLs manuais para {source.value}")
                    urls.extend(manual_urls)
                
                # Remove duplicatas mantendo a ordem
                urls = list(dict.fromkeys(urls))
                logger.info(f"Total de {len(urls)} URLs para {source.value}: {urls}")
                
                # Carrega documentos
                documents = await self.load_content(source, urls)
                logger.info(f"Carregados {len(documents)} documentos para {source.value}")
                all_documents.extend(documents)
            
            logger.info(f"Total de {len(all_documents)} documentos carregados")
            
            # Cria/atualiza a base unificada
            self.create_knowledge_base(all_documents)
                
        except Exception as e:
            logger.error(f"Erro ao atualizar bases: {e}")

    def query(self, question: str, sources: Optional[List[KnowledgeSource]] = None, k: int = 3) -> str:
        try:
            if not self.vectorstore:
                return "Base de conhecimento não inicializada."

            # Log para debug
            logging.info(f"Consultando base de conhecimento para: {question}")
            if sources:
                source_values = [s.value for s in sources]
                logging.info(f"Buscando apenas nas fontes: {source_values}")

            # Fazer busca híbrida já com filtro de sources
            docs = self.hybrid_search(question, k=k, sources=sources)
            logging.info(f"Encontrados {len(docs)} documentos")

            if not docs:
                return "Nenhuma informação relevante encontrada."

            # Log detalhado dos documentos encontrados
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'desconhecida')
                logging.info(f"Documento {i+1}:")
                logging.info(f"Fonte: {source}")
                logging.info(f"Conteúdo: {doc.page_content[:100]}...")

            # Formatar resposta
            responses = []
            for doc in docs:
                source = doc.metadata.get('source', 'desconhecida')
                emoji = {
                    'news': '📰',
                    'system': '⚙️',
                    'faq': '❓',
                    'download': '⬇️',
                    'vip': '👑',
                    'shop': '🛍️',
                    'recharge': '💰',
                    'main': '🏠'
                }.get(source, '📄')
                
                url = doc.metadata.get('url', '')
                date = doc.metadata.get('post_date', 'Data não disponível')
                response = f"{emoji} [{source.title()}] - {date}\n{doc.page_content}\nFonte: {url if url else 'Não disponível'}"
                responses.append(response)

            return "\n\n".join(responses)
                
        except Exception as e:
            logging.error(f"Erro na consulta: {e}")
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