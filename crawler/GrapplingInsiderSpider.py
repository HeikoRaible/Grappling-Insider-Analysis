import logging
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup


class GrapplingInsiderSpider(scrapy.Spider):
    name = "GrapplingInsider Spider"
    allowed_domains = ["grapplinginsider.com"]
    start_urls = ["https://grapplinginsider.com/news/"]

    def parse(self, response):
        # print(response.url)

        # only crawl on grapplinginsider.com and only crawl articles
        if "grapplinginsider.com" in response.url and "/news/" not in response.url:
            # print(response.xpath("//span[@class='cat-links']/a").extract())
            # print(response.xpath("//a[@rel='category tag']").extract())
            categories_xpath = response.xpath("//a[@rel='category tag']")
            categories = [BeautifulSoup(category_html, "html.parser").a.string.strip() for category_html in categories_xpath.extract()]
            # print(categories)

            # print(response.xpath("//header[@class='entry-header']/h1").extract())
            # print(response.xpath("//header/h1[@class='entry-title']").extract())
            # print(response.xpath("//header/h1[@itemprop='headline']").extract())
            title_xpath = response.xpath("//header/h1[@itemprop='headline']")
            title = BeautifulSoup(title_xpath.extract_first(), "html.parser").h1.string.strip()
            # print(title)

            # print(response.xpath("//div[@class='entry-content clearfix']").extract())
            # print(response.xpath("//div[@itemprop='articleBody']").extract())
            text_xpath = response.xpath("//div[@itemprop='articleBody']")
            text = BeautifulSoup(text_xpath.extract_first(), "html.parser").get_text().strip()
            # print(text)

            link_extractor = LinkExtractor()
            links = link_extractor.extract_links(response)
            external_links = [{'url': link.url, 'text': link.text.strip()} for link in links if "grapplinginsider.com" not in link.url]
            # print(external_links)

            # print("YIELD", {'url': response.url, 'categories': categories, 'title': title, 'text': text, "external_links": external_links})
            yield {'url': response.url, 'categories': categories, 'title': title, 'text': text, "external_links": external_links}

        # next articles
        for article in set(response.xpath("//div[@class='layer-content']//a/@href").extract()):
            # print("ARTICLE", article)
            yield scrapy.Request(article)

        # next page
        next_page = response.xpath("//a[@class='next page-numbers']/@href").extract_first()
        if next_page:
            # print("NEXT PAGE", next_page)
            yield scrapy.Request(next_page)
        # print()


if __name__ == "__main__":
    logging.getLogger('scrapy').propagate = False

    c = CrawlerProcess({
        'USER_AGENT': 'HochschuleDarmstadt-TextWebMining',
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'GrapplingInsider.csv',
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': True,
        'HTTPCACHE_ENABLED': True
    })
    c.crawl(GrapplingInsiderSpider)
    c.start()
