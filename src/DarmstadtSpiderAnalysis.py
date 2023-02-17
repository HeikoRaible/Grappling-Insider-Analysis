import os
import tld
import pandas as pd
from matplotlib import pyplot as plt
from urllib.parse import urlparse

import networkx as nx
from pyvis.network import Network


class DarmstadtSpiderAnalyser:
    def __init__(self):
        self.data_path = os.path.join("..", "daten")
        self.output_path = os.path.join("..", "output")
        darmstadt_spider_csv_path = os.path.join(self.data_path, "DarmstadtSpider.csv")
        self.data = pd.read_csv(darmstadt_spider_csv_path)
        #print("data:\n", self.data.head(), self.data.columns, "\n")

        # clean up links (url only domain, from domain + path)
        self.data["cleaned_url"] = self.data.url.apply(lambda link: urlparse(link).netloc.replace("www.", ""))
        self.data["cleaned_from"] = self.data["from"].apply(lambda link: (urlparse(link).netloc.replace("www.", "") + urlparse(link).path))

    def top_level_domains(self):
        """ extract tlds and create a barplot of counts """
        tlds = self.data.url.apply(tld.get_tld, fail_silently=True)
        #print("tlds:\n", tlds.head(), "\n")

        tld_counts = tlds.value_counts()
        #print("tld_counts:\n", tld_counts, "\n")

        # barplot
        for i, (tld_name, tld_count) in enumerate(tld_counts.items()):
            plt.bar(x=i, height=tld_count, label=tld_name)
        plt.title("DarmstadtSpider tlds")
        plt.xticks(range(len(tld_counts)), tld_counts.index)
        plt.ylabel("count")
        plt.show()
        #plt.savefig(os.path.join(self.output_path, "DarmstadtSpiderTLDs.png"))

    def links(self):
        # print froms grouped by url
        grouped_by_url = self.data.groupby(by="cleaned_url")
        for name, group in grouped_by_url:
            links = group["cleaned_from"].value_counts()
            # only show
            if len(links) > 1:
                print(name)
                print(links)
                print()

    def network(self):
        G = nx.from_pandas_edgelist(self.data, source="cleaned_from", target="cleaned_url")

        # networkx
        # nx.draw_circular(G)
        # plt.show()
        # plt.savefig(os.path.join(self.output_path, "DarmstadtSpiderLinks.png"))

        # pyvis
        net = Network()
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        net.save_graph(os.path.join(self.output_path, "DarmstadtSpider.html"))


if __name__ == "__main__":
    darmstadt_spider_analyser = DarmstadtSpiderAnalyser()

    # darmstadt_spider_analyser.top_level_domains()
    # darmstadt_spider_analyser.links()
    # darmstadt_spider_analyser.network()
