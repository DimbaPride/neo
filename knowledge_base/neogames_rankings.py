import os
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CharacterClass(Enum):
    GUERREIRO = (1, "GU", "Guerreiro")
    DUELISTA = (2, "DU", "Duelista")
    MAGO = (3, "MA", "Mago")
    ARQUEIRO_ARCANO = (4, "AA", "Arqueiro Arcano")
    GUARDIAO_ARCANO = (5, "GA", "Guardião Arcano")
    ESPADACHIM_ARCANO = (6, "EA", "Espadachim Arcano")
    GLADIADOR = (7, "GL", "Gladiador")
    ATIRADOR = (8, "AT", "Atirador")
    MAGO_NEGRO = (9, "MN", "Mago Negro")
    
    def __init__(self, id: int, abbr: str, full: str):
        self.id = id
        self.abbr = abbr
        self.full = full
        
    @classmethod
    def get_by_id(cls, id: int) -> Optional['CharacterClass']:
        for c in cls:
            if c.value[0] == id:
                return c
        return None

    @classmethod
    def get_by_abbr(cls, abbr: str) -> Optional['CharacterClass']:
        abbr = abbr.upper()
        for c in cls:
            if c.abbr == abbr:
                return c
        return None

class RankingType(Enum):
    POWER = "power"
    WAR = "war"
    GUILD = "guild"
    MEMORIAL = "memorial"

@dataclass
class RankingEntry:
    position: int
    name: str
    class_name: Optional[str] = None
    power: Optional[int] = None
    score: Optional[int] = None
    guild: Optional[str] = None
    timestamp: float = 0.0

class NeoGamesRankings:
    def __init__(self, base_dir: str = "knowledge_base/ranking"):
        self.base_dir = base_dir
        self.base_url = "https://www.neogames.online/ranking"
        self.embeddings = OpenAIEmbeddings()
        self.update_interval = 300
        self._monitor_task = None
        self._cached_data = {}
        
        self._create_directories()

    def _create_directories(self):
        for ranking in RankingType:
            path = os.path.join(self.base_dir, ranking.value)
            os.makedirs(path, exist_ok=True)
            if ranking == RankingType.POWER:
                os.makedirs(os.path.join(path, "general"), exist_ok=True)
                for cc in CharacterClass:
                    os.makedirs(os.path.join(path, cc.abbr.lower()), exist_ok=True)
            logger.info(f"Diretório criado: {path}")

    def get_power_ranking_urls(self) -> List[str]:
        urls = [f"{self.base_url}/power"]
        for cc in CharacterClass:
            urls.append(f"{self.base_url}/power?classId={cc.id}")
        return urls

    async def load_documents(self, url: str) -> List[Document]:
        """Carrega documentos e salva em JSON primeiro"""
        try:
            logger.info(f"Carregando dados de: {url}")
            
            # Carrega documentos com DoclingLoader
            loader = DoclingLoader(file_path=url)
            docs = await asyncio.to_thread(loader.load)
            
            if not isinstance(docs, list):
                docs = [docs]
                
            # Lista para armazenar jogadores
            players = []
            
            # Processa cada documento
            for doc in docs:
                lines = doc.page_content.split("\n")
                for line in lines:
                    if "LeaderBoard emblem1" in line or (", 2 =" in line and ", 3 =" in line):
                        parts = line.split(".")
                        player = {}
                        
                        for part in parts:
                            if "=" not in part:
                                continue
                            
                            key, value = part.split("=")
                            key = key.strip()
                            value = value.strip()
                            
                            if "2 =" in key and value != ".":  # Nome
                                player["nome"] = value
                            elif "3 =" in key and value != ".":  # Guilda
                                player["guilda"] = value
                            elif "6 =" in key and value != ".":  # Poder
                                try:
                                    poder = value.replace(",", "").replace(".", "")
                                    if poder.isdigit():
                                        player["poder"] = int(poder)
                                except:
                                    continue
                        
                        if "nome" in player and "poder" in player:
                            players.append(player)
            
            # Ordena por poder
            players.sort(key=lambda x: x.get("poder", 0), reverse=True)
            
            # Salva em JSON (para debug/backup)
            json_path = url.split("/")[-1].replace("?", "_") + ".json"
            json_path = os.path.join(self.base_dir, json_path)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(players, f, indent=2, ensure_ascii=False)
                
            # Converte para documentos
            docs = []
            for i, player in enumerate(players, 1):
                text = (
                    f"Rank: {i}\n"
                    f"Player: {player['nome']}\n"
                    f"Guilda: {player.get('guilda', 'Sem guilda')}\n"
                    f"Poder: {player['poder']:,}"
                )
                docs.append(Document(page_content=text))
                
            logger.info(f"Carregados {len(docs)} jogadores de {url}")
            logger.info(f"Dados salvos em {json_path}")
            
            # Mostra exemplos dos primeiros jogadores
            for doc in docs[:3]:
                logger.info(f"Exemplo:\n{doc.page_content}")
                
            return docs
            
        except Exception as e:
            logger.error(f"Erro ao carregar {url}: {str(e)}")
            return []



    def _parse_ranking_data(self, content: str) -> List[Dict]:
        """
        Parseia o conteúdo do ranking para um formato estruturado
        """
        try:
            players = []
            current_player = {}
            
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if not line or "Copyright" in line or "Rankings" in line:
                    continue
                    
                # Procura por entradas do tipo "LeaderBoard emblem"
                if "LeaderBoard emblem" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "=" in part:
                            key_value = part.split("=")
                            key = key_value[0].strip()
                            value = key_value[1].strip()
                            
                            # Identifica o tipo de dado
                            if key.endswith("2"):  # Nome
                                if current_player:
                                    players.append(current_player)
                                current_player = {"nome": value}
                            elif key.endswith("3"):  # Guilda
                                current_player["guilda"] = value
                            elif key.endswith("4"):  # Poder de Ataque
                                current_player["ataque"] = int(value.replace(",", "").replace(".", ""))
                            elif key.endswith("5"):  # Poder de Defesa
                                current_player["defesa"] = int(value.replace(",", "").replace(".", ""))
                            elif key.endswith("6"):  # Poder Total
                                current_player["poder_total"] = int(value.replace(",", "").replace(".", ""))
            
            if current_player:
                players.append(current_player)
                
            return players
        except Exception as e:
            logger.error(f"Erro ao parsear dados do ranking: {e}")
            return []

    async def process_documents(self, docs: List[Document], out_dir: str):
        """Processa documentos mantendo backup em JSON"""
        if not docs:
            return
            
        try:
            os.makedirs(out_dir, exist_ok=True)
            
            # Salva em JSON (para backup/debug)
            json_path = os.path.join(out_dir, "data.json")
            json_data = [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Salva no vectorstore
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            vectorstore.save_local(out_dir)
            
            logger.info(f"Salvos {len(docs)} documentos em {out_dir}")
            logger.info(f"Backup em JSON salvo em {json_path}")
            
        except Exception as e:
            logger.error(f"Erro ao processar documentos: {str(e)}")
            raise

    async def process_ranking(self, ranking: RankingType):
        """Processa os rankings com melhor validação de conteúdo"""
        if ranking == RankingType.POWER:
            urls = self.get_power_ranking_urls()
            docs_by_category: Dict[str, List[Document]] = {"general": []}
            
            # Inicializa listas para cada classe
            for cc in CharacterClass:
                docs_by_category[cc.abbr.lower()] = []
                
            for url in urls:
                try:
                    logger.info(f"Carregando dados de: {url}")
                    loader = DoclingLoader(file_path=url)
                    docs = await asyncio.to_thread(loader.load)
                    
                    if not isinstance(docs, list):
                        docs = [docs]
                    
                    # Loga o conteúdo para debug
                    for doc in docs:
                        logger.info(f"Conteúdo carregado: {doc.page_content[:200]}...")
                    
                    # Valida e filtra documentos
                    valid_docs = []
                    for doc in docs:
                        content = doc.page_content.strip()
                        if content and len(content) > 0 and not content == "Poder de Combate":
                            valid_docs.append(doc)
                    
                    if "classId=" in url:
                        try:
                            class_id_str = url.split("classId=")[1].split("&")[0]
                            class_id = int(class_id_str)
                            cc = CharacterClass.get_by_id(class_id)
                            if cc:
                                category = cc.abbr.lower()
                                docs_by_category[category].extend(valid_docs)
                                logger.info(f"Adicionados {len(valid_docs)} documentos para {category}")
                        except Exception as e:
                            logger.error(f"Erro ao processar URL {url}: {e}")
                    else:
                        docs_by_category["general"].extend(valid_docs)
                        logger.info(f"Adicionados {len(valid_docs)} documentos para general")
                        
                except Exception as e:
                    logger.error(f"Erro ao carregar {url}: {e}")
                    continue
            
            # Processa e salva cada categoria
            for category, docs in docs_by_category.items():
                if docs:
                    out_dir = os.path.join(self.base_dir, RankingType.POWER.value, category)
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        
                        # Divide em chunks menores
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=400,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", " "]
                        )
                        
                        chunks = splitter.split_documents(docs)
                        if chunks:
                            # Adiciona metadados aos chunks
                            for chunk in chunks:
                                chunk.metadata["source"] = category
                                chunk.metadata["type"] = "power_ranking"
                            
                            # Salva o vectorstore
                            vectorstore = FAISS.from_documents(chunks, self.embeddings)
                            vectorstore.save_local(out_dir)
                            logger.info(f"Vectorstore salvo em {out_dir} com {len(chunks)} chunks")
                        else:
                            logger.warning(f"Nenhum chunk válido gerado para {category}")
                            
                    except Exception as e:
                        logger.error(f"Erro ao processar documentos para {category}: {e}")
                else:
                    logger.warning(f"Nenhum documento válido para categoria: {category}")
        else:
            # Processamento para outros tipos de ranking continua igual
            url = f"{self.base_url}/{ranking.value}"
            docs = await self.load_documents(url)
            if docs:
                out_dir = os.path.join(self.base_dir, ranking.value)
                await self.process_documents(docs, out_dir)

    def query(
        self,
        question: str,
        ranking_types: Optional[List[RankingType]] = None,
        k: int = 3,
        class_abbr: Optional[str] = None
    ) -> str:
        """Consulta rankings de forma simples"""
        if ranking_types is None:
            ranking_types = list(RankingType)

        responses = []
        for ranking in ranking_types:
            try:
                # Define o caminho
                if ranking == RankingType.POWER:
                    subfolder = "general"
                    if class_abbr:
                        class_abbr = class_abbr.lower()
                        if class_abbr in [cc.abbr.lower() for cc in CharacterClass]:
                            subfolder = class_abbr
                    store_path = os.path.join(self.base_dir, ranking.value, subfolder)
                else:
                    store_path = os.path.join(self.base_dir, ranking.value)

                # Verifica se existe
                if not os.path.exists(store_path):
                    continue

                # Carrega e busca
                vectorstore = FAISS.load_local(store_path, self.embeddings, allow_dangerous_deserialization=True)
                docs = vectorstore.similarity_search(question, k=k)

                # Formata resposta
                if docs:
                    header = f"[{ranking.value.upper()}"
                    if ranking == RankingType.POWER and class_abbr:
                        header += f" - {subfolder}"
                    header += "]"
                    
                    content = []
                    for doc in docs:
                        if doc.page_content.strip():
                            content.append(doc.page_content.strip())
                    
                    if content:
                        responses.append(f"{header}\n" + "\n\n".join(content))

            except Exception as e:
                logger.error(f"Erro consultando {ranking.value}: {e}")
                continue

        if responses:
            return "\n\n".join(responses)
        return "Ranking temporariamente indisponível. Por favor, tente novamente mais tarde."

    async def initialize(self):
        """Inicializa e processa todos os rankings."""
        try:
            logger.info("Iniciando processamento de rankings...")
            for ranking in RankingType:
                logger.info(f"Processando ranking: {ranking.value}")
                await self.process_ranking(ranking)
            logger.info("Processamento de rankings concluído.")
        except Exception as e:
            logger.error(f"Erro na inicialização: {e}")
            raise

    def __del__(self):
        """Limpa recursos ao destruir a instância."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

# Exporta as classes principais
__all__ = ["NeoGamesRankings", "RankingType", "CharacterClass", "RankingEntry"]