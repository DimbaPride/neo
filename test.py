import asyncio
from playwright.async_api import Page
from langchain_community.document_loaders import PlaywrightURLLoader
from knowledge_base.neogames_knowledge import NeoGamesKnowledge, KnowledgeSource

class CustomPlaywrightURLLoader(PlaywrightURLLoader):
    async def _get_page_content(self, page: Page) -> str:
        # Aguarda o carregamento inicial da página
        await page.wait_for_load_state("networkidle")
        
        # Seleciona e clica em todos os botões de dropdown
        buttons = await page.query_selector_all("button[aria-controls]")
        for button in buttons:
            if (await button.get_attribute("aria-expanded")) != "true":
                await button.click()
                await asyncio.sleep(1)  # Aguarda 1 segundo entre os cliques
        
        # Aguarda que um elemento que contenha o conteúdo das FAQs esteja presente.
        # No exemplo, supomos que o conteúdo expandido esteja dentro de um <article class="prose">.
        try:
            await page.wait_for_selector("article.prose", timeout=10000)
        except Exception as e:
            print("Elemento específico de conteúdo FAQ não encontrado:", e)
        
        # Aumenta o tempo de espera para garantir que o conteúdo seja carregado
        await asyncio.sleep(5)
        return await page.content()

async def test_load_faq_expanded():
    knowledge = NeoGamesKnowledge()
    faq_urls = [f"{knowledge.base_url}/faq"]

    loader = CustomPlaywrightURLLoader(
        urls=faq_urls,
        remove_selectors=[
            "nav", "footer", "header", ".modal", "script", "noscript", "style"
        ]
    )
    
    print("Testando load_content para FAQ (com dropdowns expandidos)...")
    documents = await loader.aload()
    if not documents:
        print("Nenhum documento carregado para FAQ.")
    else:
        for doc in documents:
            print("Documento (FAQ):")
            print("Conteúdo:", doc.page_content[:1000])
            print("Metadata:", doc.metadata)
            print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_load_faq_expanded())
