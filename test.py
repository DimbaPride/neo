from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# URL do ranking de poder
RANKING_URL = "https://www.neogames.online/ranking/power"

def fetch_ranking_data_with_playwright(url):
    """
    Usa Playwright para carregar a página e extrair o conteúdo HTML.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Executa em modo headless
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")  # Aguarda até que a página esteja totalmente carregada
        html_content = page.content()  # Obtém o HTML completo
        browser.close()
        return html_content

def parse_ranking_data(html_content):
    """
    Analisa o conteúdo HTML para extrair os dados do ranking.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Suponha que os dados do ranking estão em uma tabela com a classe 'ranking-table'
    ranking_table = soup.find('table', class_='ranking-table') or soup.find('div', class_='ranking-content')
    if not ranking_table:
        print("Nenhuma tabela ou container de ranking encontrado.")
        return []

    # Extrai os cabeçalhos da tabela (se houver)
    headers = []
    if ranking_table.find('thead'):
        headers = [header.get_text(strip=True) for header in ranking_table.find('thead').find_all('th')]

    # Extrai as linhas da tabela
    rows = []
    for row in ranking_table.find_all('tr')[1:]:  # Ignora a primeira linha (cabeçalhos)
        cells = row.find_all(['td', 'div'])  # Procura tanto por <td> quanto por <div>
        if cells:
            row_data = [cell.get_text(strip=True) for cell in cells]
            rows.append(dict(zip(headers, row_data)) if headers else row_data)

    return rows

def main():
    # Passo 1: Obter o conteúdo HTML da página usando Playwright
    print("Carregando a página com Playwright...")
    html_content = fetch_ranking_data_with_playwright(RANKING_URL)
    if not html_content:
        print("Falha ao carregar a página.")
        return

    # Passo 2: Analisar o conteúdo HTML e extrair os dados do ranking
    print("Analisando o conteúdo HTML...")
    ranking_data = parse_ranking_data(html_content)

    # Passo 3: Exibir os dados extraídos
    if ranking_data:
        print("Dados do Ranking de Poder:")
        for entry in ranking_data:
            print(entry)
    else:
        print("Nenhum dado de ranking encontrado.")

if __name__ == "__main__":
    main()