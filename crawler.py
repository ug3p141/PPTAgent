import asyncio
import os

import aiohttp
from googlesearch import search

from presentation import Presentation

ppt_topics = [
    "Business Plan",
    "Marketing Strategy",
    "Project Management",
    "Financial Analysis",
    "Product Launch",
    "Company Overview",
    "Sales Report",
    "Training and Development",
    "Industry Trends",
    "Customer Experience",
]
pptx_links = []


def get_pptx_links(topic, num_results=20):
    query = f"{topic} filetype:pptx"
    return [
        {"title": url.split("/")[-1], "url": url}
        for url in search(query, num_results=num_results)
        if url.endswith(".pptx")
    ]


async def download_file(session, url, filepath):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, "wb") as f:
                    f.write(await response.read())
                presentation = Presentation.from_file(filepath)
                if (
                    len(presentation.slides) < 6
                    or len(presentation.slides) > 50
                    or len(presentation.error_history) > len(presentation.slides) // 3
                ):
                    os.remove(filepath)
    except:
        return


async def download_pptx_files():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for topic in ppt_topics:
            links = get_pptx_links(topic)
            pptx_links.extend(links)
            for link in links:
                filepath = f"resource/crawler/{link['title'].replace(' ', '_')}"
                tasks.append(download_file(session, link["url"], filepath))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    os.makedirs("resource/crawler", exist_ok=True)
    asyncio.run(download_pptx_files())
    print(f"Downloaded {len(pptx_links)} pptx files in resource/crawler")
