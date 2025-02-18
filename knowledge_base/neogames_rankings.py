import os
import asyncio
import logging
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configuração de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# URLs base
BASE_URL = "https://www.neogames.online/ranking"
GUILD_RANKING_URL = f"{BASE_URL}/guild"
MEMORIAL_RANKING_URL = f"{BASE_URL}/memorial"
POWER_RANKING_URL = f"{BASE_URL}/power"

class CharacterClass(Enum):
    GUERREIRO = (1, "GU", "Guerreiro", "wa", "Icon WA", "Warrior")
    DUELISTA = (2, "DU", "Duelista", "bl", "Icon BL", "Blader")
    MAGO = (3, "MA", "Mago", "wz", "Icon WZ", "Wizard")
    ARQUEIRO_ARCANO = (4, "AA", "Arqueiro Arcano", "fa", "Icon FA", "Force Archer")
    GUARDIAO_ARCANO = (5, "GA", "Guardião Arcano", "fs", "Icon FS", "Force Shielder")
    ESPADACHIM_ARCANO = (6, "EA", "Espadachim Arcano", "fb", "Icon FB", "Force Blader")
    GLADIADOR = (7, "GL", "Gladiador", "gl", "Icon GL", "Gladiator")
    ATIRADOR = (8, "AT", "Atirador", "fg", "Icon FG", "Force Gunner")
    MAGO_NEGRO = (9, "MN", "Mago Negro", "dm", "Icon DM", "Dark Mage")
    
    def __init__(self, id: int, abbr: str, full_pt: str, icon: str, alt: str, full_en: str):
        self.id = id
        self.abbr = abbr
        self.full_pt = full_pt
        self.full_en = full_en
        self.icon = icon
        self.alt = alt
        
    @classmethod
    def get_by_id(cls, id: int) -> Optional['CharacterClass']:
        """Retorna a classe pelo ID"""
        for char_class in cls:
            if char_class.id == id:
                return char_class
        return None

    @classmethod
    def get_by_abbr(cls, abbr: str) -> Optional['CharacterClass']:
        """Retorna a classe pela abreviação"""
        abbr = abbr.upper()
        for char_class in cls:
            if char_class.abbr == abbr:
                return char_class
        return None
    
    @classmethod
    def get_by_icon(cls, icon_src: str) -> Optional['CharacterClass']:
        """Retorna a classe pelo ícone"""
        for char_class in cls:
            if f"icon-{char_class.icon}" in icon_src:
                return char_class
        return None

class RankingType(Enum):
    POWER = "power"
    WAR = "war"
    GUILD = "guild"
    MEMORIAL = "memorial"

# Mapeamento de nações
NATION_MAPPING = {
    'icon-procyon': {
        'name': 'Procyon',
        'name_pt': 'Procion'
    },
    'icon-capella': {
        'name': 'Capella',
        'name_pt': 'Capella'
    }
}

class NeoGamesRankings:
    def __init__(self, base_dir: str = "knowledge_base/ranking"):
        self.base_dir = base_dir
        self.embeddings = OpenAIEmbeddings()
        self._create_directories()
        
        # Headers para cada tipo de ranking
        self.power_headers = {
            'position': '#',
            'class': 'Classe',
            'name': 'Nome',
            'guild': 'Guilda',
            'attack_power': 'Poder de Ataque',
            'defense_power': 'Poder de Defesa',
            'total_power': 'Poder total',
            'nation': 'Nação'
        }
        
        self.guild_headers = {
            'position': '#',
            'name': 'Nome',
            'power': 'Poder',
            'members': 'Membros',
            'war_points': 'Pontos de Guerra',
            'war_kills': 'Abates na Guerra'
        }
        
        self.memorial_headers = {
            'position': '#',
            'character_name': 'Nome do Personagem',
            'character_class': 'Classe',
            'guild_name': 'Guild',
            'nation': 'Nação'
        }

    def _create_directories(self):
        """Cria estrutura de diretórios para os rankings."""
        for ranking in RankingType:
            path = os.path.join(self.base_dir, ranking.value)
            os.makedirs(path, exist_ok=True)
            if ranking == RankingType.POWER:
                os.makedirs(os.path.join(path, "general"), exist_ok=True)
                for cc in CharacterClass:
                    os.makedirs(os.path.join(path, cc.abbr.lower()), exist_ok=True)
            logger.info(f"Diretório criado: {path}")

    async def fetch_page_content(self, url: str, wait_selector='table', timeout=30000) -> str:
        """Busca o conteúdo de uma página usando Playwright"""
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
                )
                
                page = await context.new_page()
                page.set_default_timeout(60000)
                
                response = await page.goto(url)
                
                if not response or response.status != 200:
                    raise Exception(f"Falha ao carregar página: Status {response.status if response else 'N/A'}")
                
                await page.wait_for_selector(wait_selector, timeout=timeout)
                await page.wait_for_timeout(3000)
                
                content = await page.content()
                await browser.close()
                return content
                
            except Exception as e:
                logger.error(f"Erro ao buscar página {url}: {e}")
                if 'browser' in locals():
                    await browser.close()
                raise

    def parse_value(self, value_str: str) -> int:
        """Converte valores string para formato numérico."""
        try:
            clean_value = ''.join(c for c in value_str if c.isdigit() or c in '.,')
            clean_value = clean_value.replace('.', '').replace(',', '.')
            return int(float(clean_value))
        except:
            return 0

    def get_nation_info(self, cell_or_src) -> Dict:
        """Identifica a nação baseado na célula da tabela ou src da imagem."""
        try:
            if hasattr(cell_or_src, 'find'):
                img = cell_or_src.find('img')
                if img and 'srcset' in img.attrs:
                    srcset = img['srcset']
                    if 'icon-procyon.png' in srcset:
                        return NATION_MAPPING['icon-procyon']
                    elif 'icon-capella.png' in srcset:
                        return NATION_MAPPING['icon-capella']
            elif isinstance(cell_or_src, str):
                text_lower = cell_or_src.lower()
                if 'procyon' in text_lower:
                    return NATION_MAPPING['icon-procyon']
                elif 'capella' in text_lower:
                    return NATION_MAPPING['icon-capella']
        except Exception as e:
            logger.warning(f"Erro ao identificar nação: {e}")
        
        return {
            'name': 'Unknown',
            'name_pt': 'Desconhecida'
        }

    def parse_power_ranking(self, html_content: str) -> List[Dict]:
        """Analisa o HTML para extrair dados do ranking de power."""
        logger.info("Analisando dados do ranking de power")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        power_data = []
        
        try:
            rows = soup.find_all('tr')[1:]  # Pula o cabeçalho
            
            for position, row in enumerate(rows, 1):
                try:
                    cells = row.find_all(['td'])
                    if len(cells) >= 7:
                        # Identifica a classe
                        class_cell = cells[1]
                        class_info = None
                        
                        class_img = class_cell.find('img')
                        if class_img and 'srcset' in class_img.attrs:
                            srcset = class_img['srcset']
                            for char_class in CharacterClass:
                                if f"/ranking/icon-{char_class.icon}.png" in srcset:
                                    class_info = char_class
                                    break
                        
                        if not class_info:
                            class_info = CharacterClass.GUERREIRO
                        
                        # Identifica a nação
                        nation_cell = cells[7] if len(cells) >= 8 else None
                        nation_info = self.get_nation_info(nation_cell) if nation_cell else {
                            'name': 'Unknown',
                            'name_pt': 'Desconhecida'
                        }
                        
                        power_entry = {
                            'position': position,
                            'class': {
                                'id': class_info.id,
                                'abbr': class_info.abbr,
                                'full_pt': class_info.full_pt,
                                'full_en': class_info.full_en
                            },
                            'name': cells[2].get_text(strip=True),
                            'guild': cells[3].get_text(strip=True),
                            'attack_power': self.parse_value(cells[4].get_text(strip=True)),
                            'defense_power': self.parse_value(cells[5].get_text(strip=True)),
                            'total_power': self.parse_value(cells[6].get_text(strip=True)),
                            'nation': {
                                'en': nation_info['name'],
                                'pt': nation_info['name_pt']
                            }
                        }
                        
                        power_data.append(power_entry)
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar power {position}: {e}")
                    continue
            
            return power_data
            
        except Exception as e:
            logger.error(f"Erro ao analisar ranking de power: {e}")
            raise

    def save_ranking_data(self, data: List[Dict], ranking_type: str, class_id: Optional[int] = None):
        """Salva os dados do ranking em JSON e cria índices FAISS."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define o diretório de saída
            if ranking_type == 'power' and class_id:
                class_info = CharacterClass.get_by_id(class_id)
                if not class_info:
                    raise ValueError(f"Classe ID {class_id} não encontrada")
                subfolder = class_info.abbr.lower()
            else:
                subfolder = "general"
                
            out_dir = os.path.join(self.base_dir, ranking_type, subfolder)
            os.makedirs(out_dir, exist_ok=True)
            
            # Nome do arquivo JSON
            json_filename = f"ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = os.path.join(out_dir, json_filename)
            
            # Prepara dados para JSON
            output_data = {
                'timestamp': timestamp,
                'total_entries': len(data),
                'rankings': data
            }
            
            # Adiciona info da classe se necessário
            if class_id:
                class_info = CharacterClass.get_by_id(class_id)
                output_data['class_info'] = {
                    'id': class_info.id,
                    'abbr': class_info.abbr,
                    'full_pt': class_info.full_pt,
                    'full_en': class_info.full_en,
                    'icon': class_info.icon
                }
            
            # Salva JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dados JSON salvos em: {json_path}")
            
            # Prepara documentos para FAISS
            docs = []
            for entry in data:
                if ranking_type == 'power':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Player: {entry['name']}\n"
                        f"Classe: {entry['class']['full_pt']} ({entry['class']['abbr']})\n"
                        f"Guild: {entry['guild']}\n"
                        f"Poder Total: {entry['total_power']:,}\n"
                        f"Poder de Ataque: {entry['attack_power']:,}\n"
                        f"Poder de Defesa: {entry['defense_power']:,}"
                    )
                elif ranking_type == 'guild':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Guild: {entry['name']}\n"
                        f"Poder: {entry['power']:,}\n"
                        f"Membros: {entry['members']}\n"
                        f"Pontos de Guerra: {entry['war_points']:,}\n"
                        f"Abates na Guerra: {entry['war_kills']:,}"
                    )
                elif ranking_type == 'memorial':
                    content = (
                        f"Rank: {entry['position']}\n"
                        f"Player: {entry['character_name']}\n"
                        f"Classe: {entry['character_class']['pt']} ({entry['character_class']['short']})\n"
                        f"Guild: {entry['guild_name']}\n"
                        f"Nação: {entry['nation']['pt']}"
                    )
                else:
                    content = str(entry)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'timestamp': timestamp,
                        'ranking_type': ranking_type,
                        'position': entry['position']
                    }
                )
                docs.append(doc)
            
            # Cria e salva vectorstore
            if docs:
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                vectorstore.save_local(out_dir)
                logger.info(f"Índice FAISS salvo em: {out_dir}")
            
            # Mostra exemplo dos primeiros colocados
            # Mostra exemplo dos primeiros colocados
            if data:
                class_name = CharacterClass.get_by_id(class_id).full_pt if class_id else 'Geral'
                logger.info(f"\nTop 3 {class_name}:")
                for entry in data[:3]:
                    if ranking_type == 'power':
                        logger.info(
                            f"#{entry['position']}: {entry['name']} "
                            f"({entry['class']['full_pt']}) - "
                            f"Guild: {entry['guild']} - "
                            f"Power Total: {entry['total_power']:,}"
                        )
                    elif ranking_type == 'memorial':
                        logger.info(
                            f"#{entry['position']}: {entry['character_name']} "
                            f"({entry['character_class']['pt']}) - "
                            f"Guild: {entry['guild_name']}"
                        )
                    else:  # guild
                        logger.info(
                            f"#{entry['position']}: {entry['name']} - "
                            f"Poder: {entry['power']:,} - "
                            f"Membros: {entry['members']}"
                        )
                        
        except Exception as e:
            logger.error(f"Erro ao salvar ranking: {e}")
            raise

    async def process_ranking(self, ranking: RankingType):
        """Processa os rankings de forma assíncrona"""
        try:
            if ranking == RankingType.POWER:
                # Processa ranking geral
                html_content = await self.fetch_page_content(POWER_RANKING_URL)
                power_data = self.parse_power_ranking(html_content)
                
                if power_data:
                    self.save_ranking_data(power_data, 'power')
                
                # Processa rankings por classe
                for char_class in CharacterClass:
                    url = f"{POWER_RANKING_URL}?classId={char_class.id}"
                    logger.info(f"Processando ranking de {char_class.full_pt}")
                    
                    html_content = await self.fetch_page_content(url)
                    power_data = self.parse_power_ranking(html_content)
                    
                    if power_data:
                        self.save_ranking_data(power_data, 'power', char_class.id)
                    await asyncio.sleep(1)
                    
            elif ranking == RankingType.GUILD:
                html_content = await self.fetch_page_content(GUILD_RANKING_URL)
                guild_data = self.parse_guild_ranking(html_content)
                if guild_data:
                    self.save_ranking_data(guild_data, 'guild')
                    
            elif ranking == RankingType.MEMORIAL:
                html_content = await self.fetch_page_content(
                    MEMORIAL_RANKING_URL,
                    wait_selector='div.grid.grid-cols-1'
                )
                memorial_data = self.parse_memorial_ranking(html_content)
                if memorial_data:
                    self.save_ranking_data(memorial_data, 'memorial')
            
        except Exception as e:
            logger.error(f"Erro ao processar ranking {ranking}: {e}")
            raise

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

    def query(
        self,
        question: str,
        ranking_types: Optional[List[RankingType]] = None,
        k: int = 3,
        class_abbr: Optional[str] = None
    ) -> str:
        """Consulta rankings usando FAISS"""
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
                        header += f" - {class_abbr.upper()}"
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
