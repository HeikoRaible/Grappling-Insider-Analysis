import os
import tld
import ast
import pandas as pd
from matplotlib import pyplot as plt
from urllib.parse import urlparse

import networkx as nx
from pyvis.network import Network


class GrapplingInsiderAnalyser:
    def __init__(self):
        self.data_path = os.path.join("..", "daten")
        self.output_path = os.path.join("..", "output")
        grappling_insider_csv_path = os.path.join(self.data_path, "GrapplingInsider.csv")
        self.data = pd.read_csv(grappling_insider_csv_path)
        # print("data:\n", self.data.head(), self.data.columns, "\n")

        # convert to old data format
        data = []
        for row in self.data.iterrows():
            for link in ast.literal_eval(row[1][4]):
                data.append([row[1][0], link["url"], link["text"]])

        self.formatted_data = pd.DataFrame(data, columns=["from", "url", "text"])
        # print(self.formatted_data.head(), self.formatted_data.columns)

        # clean up links (url only domain, from domain + path)
        self.formatted_data["cleaned_url"] = self.formatted_data.url.apply(lambda link: urlparse(link).netloc.replace("www.", "").replace("www1.", ""))
        self.formatted_data["cleaned_from"] = self.formatted_data["from"].apply(lambda link: (urlparse(link).netloc.replace("www.", "") + urlparse(link).path))

        # filter bad urls
        # print(self.formatted_data.shape)
        self.formatted_data = self.formatted_data[~self.formatted_data.cleaned_url.str.contains(" ")]
        self.formatted_data = self.formatted_data[~self.formatted_data.cleaned_url.str.contains("\(")]
        self.formatted_data = self.formatted_data[~self.formatted_data.cleaned_url.isin(["facebook.com", "m.facebook.com", "instagram.com", "youtube.com", "m.youtube.com", "youtu.be", "twitter.com", "t.co", "ezoic.com", "yelp.com", "linkedin.com"])]
        # print(self.formatted_data.shape)

    def get_max_lens(self):
        max_url = 0
        max_category = 0
        max_title = 0
        max_text = 0
        max_link = 0
        max_link_text = 0
        for _, row in self.data.iterrows():
            max_url = len(row.url) if len(row.url) > max_url else max_url
            for category in row.categories.split(","):
                max_category = len(category) if len(category) > max_category else max_category
            max_title = len(row.title) if len(row.title) > max_title else max_title
            max_text = len(row.text) if len(row.text) > max_text else max_text
            for external_link in ast.literal_eval(row.external_links):
                max_link = len(external_link["url"]) if len(external_link["url"]) > max_link else max_link
                max_link_text = len(external_link["text"]) if len(
                    external_link["text"]) > max_link_text else max_link_text
        print(f"url: {max_url}")
        print(f"category: {max_category}")
        print(f"title: {max_title}")
        print(f"text: {max_text}")
        print(f"link: {max_link}")
        print(f"link_text: {max_link_text}")

    def text_filter(self):
        data = self.data[self.data.text.str.contains("Ryan Hall")]
        print(data.url.unique())

    def top_level_domains(self):
        """ extract tlds and create a barplot of counts """
        tlds = self.formatted_data.url.apply(tld.get_tld, fail_silently=True)
        tld_counts = tlds.value_counts()
        print("tld_counts:\n", tld_counts, "\n")

        # barplot
        for i, (tld_name, tld_count) in enumerate(tld_counts.items()):
            plt.bar(x=i, height=tld_count, label=tld_name)
        plt.title("GrapplingInsider tlds")
        plt.xticks(range(len(tld_counts)), tld_counts.index, rotation=45)
        plt.ylabel("count")
        # plt.show()
        plt.savefig(os.path.join(self.output_path, "GrapplingInsiderTLDs.png"))

    def links(self):
        # print froms grouped by url
        grouped_by_url = self.formatted_data.groupby(by="cleaned_url")
        for name, group in grouped_by_url:
            links = group["cleaned_from"].value_counts()
            # only show
            if len(links) > 1:
                print(name)
                print(links)
                print()

    def network(self):
        G = nx.from_pandas_edgelist(self.formatted_data, source="cleaned_from", target="cleaned_url")

        # networkx
        # nx.draw_circular(G)
        # plt.show()
        # plt.savefig(os.path.join(self.output_path, "GrapplingInsiderLinks.png"))

        # pyvis
        net = Network()
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        net.save_graph(os.path.join(self.output_path, "GrapplingInsider.html"))


if __name__ == "__main__":
    grappling_insider_analyser = GrapplingInsiderAnalyser()

    # grappling_insider_analyser.top_level_domains()
    # grappling_insider_analyser.links()
    # grappling_insider_analyser.network()

    # grappling_insider_analyser.text_filter()
    # grappling_insider_analyser.get_max_lens()
