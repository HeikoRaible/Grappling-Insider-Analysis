import os
import pandas as pd
import matplotlib.pyplot as plt

from hdbcli import dbapi
from ast import literal_eval
from wordcloud import WordCloud
from math import log10, sqrt


class HanaHandler:
    def __init__(self):
        connection = dbapi.connect('192.168.56.102', 39041, 'SYSTEM', 'Password1')
        self.cursor = connection.cursor()

        self.data_path = os.path.join("..", "daten")
        self.output_path = os.path.join("..", "output")

    def create_tables(self):
        self.cursor.execute('CREATE TABLE GRAPPLING_INSIDER_CONTENT( url VARCHAR(300) PRIMARY KEY, title VARCHAR(300), text NCLOB MEMORY THRESHOLD 1000 )')
        self.cursor.execute('CREATE TABLE GRAPPLING_INSIDER_CATEGORIES( url VARCHAR(300), category VARCHAR(30) )')
        self.cursor.execute('CREATE TABLE GRAPPLING_INSIDER_EXTERNAL_LINKS( url VARCHAR(300), external_link_url NCLOB MEMORY THRESHOLD 1000, external_link_text VARCHAR(700) )')

    def insert_data(self):
        # default insert format
        sql_insert_grappling_insider_content = "insert into GRAPPLING_INSIDER_CONTENT (url, title, text) VALUES (?, ?, ?)"
        sql_insert_grappling_insider_categories = "insert into GRAPPLING_INSIDER_CATEGORIES (url, category) VALUES (?, ?)"
        sql_insert_grappling_insider_external_links = "insert into GRAPPLING_INSIDER_EXTERNAL_LINKS (url, external_link_url, external_link_text) VALUES (?, ?, ?)"

        # go through data
        data_path = os.path.join("..", "data")
        grappling_insider_csv_path = os.path.join(data_path, "GrapplingInsider.csv")
        data = pd.read_csv(grappling_insider_csv_path)
        for _, row in data.iterrows():
            self.cursor.execute(sql_insert_grappling_insider_content, (row["url"], row["title"], row["text"]))
            for category in row["categories"].split(","):
                self.cursor.execute(sql_insert_grappling_insider_categories, (row["url"], category))
            for external_link in literal_eval(row["external_links"]):
                self.cursor.execute(sql_insert_grappling_insider_external_links, (row["url"], external_link["url"], external_link["text"]))

    def create_index(self, config):
        command = f"""
        CREATE FULLTEXT INDEX "GRAPPLING_INSIDER_INDEX"
        ON "SYSTEM"."GRAPPLING_INSIDER_CONTENT" ("TEXT")
        CONFIGURATION '{config}'
        ASYNC LANGUAGE DETECTION ('en')
        TEXT ANALYSIS ON
        """
        self.cursor.execute(command)

    def drop_index(self):
        command = """
        DROP FULLTEXT INDEX "GRAPPLING_INSIDER_INDEX"
        """
        self.cursor.execute(command)

    def index_analysis(self):
        # lexicon length
        command = """
        SELECT COUNT(DISTINCT(TA_TOKEN))
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        """
        self.cursor.execute(command)
        print("lexicon length:", [x[0] for x in self.cursor.fetchall()][0], "tokens")

        # filtered lexicon length
        command = """
        SELECT COUNT(DISTINCT(TA_TOKEN))
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        WHERE TA_TYPE='adjective' OR TA_TYPE='verb' OR TA_TYPE='noun' OR TA_TYPE='proper name'
        """
        self.cursor.execute(command)
        print("filtered lexicon length:", [x[0] for x in self.cursor.fetchall()][0], "tokens")

        # average document length
        command = """
        SELECT COUNT(TA_TOKEN)
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        GROUP BY URL
        """
        self.cursor.execute(command)
        tokens_per_document = [x[0] for x in self.cursor.fetchall()]
        print("average document length:", round(sum(tokens_per_document)/len(tokens_per_document)), "tokens")

        # average sentence length
        command = """
        SELECT COUNT(TA_TOKEN)
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        GROUP BY URL, TA_SENTENCE
        """
        self.cursor.execute(command)
        tokens_per_sentence = [x[0] for x in self.cursor.fetchall()]
        print("average sentence length:", round(sum(tokens_per_sentence) / len(tokens_per_sentence)), "tokens")

        # lemmatization?

    def wordcloud(self):
        for mode in ["top", "bottom"]:
            command = f"""
            SELECT top 20 TA_TOKEN, count(*)
            FROM "$TA_GRAPPLING_INSIDER_INDEX"
            WHERE TA_TYPE=\'noun\'
            GROUP BY TA_TOKEN
            ORDER BY count(*) {"desc" if mode=="top" else "asc"}
            """
            self.cursor.execute(command)

            tmpDict = {}
            for row in self.cursor:
                tmpDict[row[0]] = row[1]
            print(tmpDict)

            # barplot
            plt.barh(list(tmpDict.keys()), list(tmpDict.values()))
            plt.gca().invert_yaxis()
            plt.title(f"{mode} 20 tokens")
            plt.savefig(os.path.join(self.output_path, f"{mode}_20_tokens.png"), bbox_inches="tight")
            plt.close()

            # wordcloud
            wordcloud = WordCloud(width=480, height=480, margin=0).generate_from_frequencies(tmpDict)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.margins(x=0, y=0)
            plt.savefig(os.path.join(self.output_path, f"{mode}_20_wordcloud.png"), bbox_inches="tight")
            plt.close()

    def word_ambiguity(self):
        command = """
        SELECT URL, TA_TOKEN, TA_TYPE
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        WHERE TA_TYPE='adjective' OR TA_TYPE='verb' OR TA_TYPE='noun'
        ORDER BY URL ASC
        """
        self.cursor.execute(command)

        # pandas df
        data = pd.DataFrame(self.cursor.fetchall(), columns=["url", "token", "pos"])

        # get pos count fraction in respective url per token
        colors = {"noun": "tab:blue", "verb": "tab:orange", "adjective": "tab:green"}
        data_dict = {}
        for url in data.url.unique():
            subdata = data[data.url == url]
            for token in subdata.token.unique():
                pos_per_token = set(subdata[subdata.token == token].pos)
                if len(pos_per_token) > 2:
                    if token not in data_dict:
                        data_dict[token] = {}
                    pos_cnts = [len(subdata[(subdata.token == token) & (subdata.pos == pos)]) for pos in pos_per_token]
                    for i, pos in enumerate(pos_per_token):
                        if pos in data_dict[token]:
                            data_dict[token][pos].append(pos_cnts[i]/sum(pos_cnts))
                        else:
                            data_dict[token][pos] = [pos_cnts[i]/sum(pos_cnts)]

        # calculate average pos count fraction per token over all urls
        for token, pos_data in data_dict.items():
            for pos, pos_values in pos_data.items():
                data_dict[token][pos] = sum(pos_values)/len(pos_values)
            # sort pos values
            data_dict[token] = dict(reversed(sorted(data_dict[token].items(), key=lambda item: item[1])))

        # sort data by deviation from pos frequency parity
        data_dict = dict(sorted(data_dict.items(), key=lambda item: sum([abs(x-0.33) for x in item[1].values()])))

        # plot data
        for i, token in enumerate(data_dict.keys()):
            total_cnt = 0
            for pos, cnt in data_dict[token].items():
                plt.bar(x=token, height=cnt, bottom=total_cnt, color=colors[pos], label=pos)
                total_cnt += cnt
        plt.axhline(y=0.333, linestyle="--", color="black")
        plt.axhline(y=0.666, linestyle="--", color="black")

        # settings
        plt.title("word ambiguity: PoS fractions per token (overall)")
        plt.xticks(rotation=90)
        plt.yticks([0.333, 0.666, 1], ["1/3", "2/3", "3/3"])

        # legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        plt.legend(*zip(*unique), loc="upper left", bbox_to_anchor=(0.99, 0.99), frameon=False)

        # show/save plot
        # plt.show()
        plt.savefig(os.path.join(self.output_path, f"word_ambiguity.png"), bbox_inches="tight")

    def word_ambiguity_per_url(self):
        command = """
        SELECT URL, TA_TOKEN, TA_TYPE
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        WHERE TA_TYPE='adjective' OR TA_TYPE='verb' OR TA_TYPE='noun'
        ORDER BY URL ASC
        """
        self.cursor.execute(command)

        # pandas df
        data = pd.DataFrame(self.cursor.fetchall(), columns=["url", "token", "pos"])

        # plot data
        colors = {"noun": "tab:blue", "verb": "tab:orange", "adjective": "tab:green"}
        data_dict = {}
        for url in data.url.unique():
            data_dict[url] = {}
            subdata = data[data.url == url]
            for token in subdata.token.unique():
                pos_per_token = set(subdata[subdata.token == token].pos)
                if len(pos_per_token) > 1:
                    data_dict[url][token] = {}
                    for pos in pos_per_token:
                        data_dict[url][token][pos] = len(subdata[(subdata.token == token) & (subdata.pos == pos)])

            # sort data
            data_dict[url] = dict(reversed(sorted(data_dict[url].items(), key=lambda item: sum(item[1].values()))))

            # plot data per url
            for token in data_dict[url].keys():
                total_cnt = 0
                for pos, cnt in data_dict[url][token].items():
                    plt.bar(x=token, height=cnt, bottom=total_cnt, color=colors[pos], label=pos)
                    total_cnt += cnt

            # settings
            plt.title("PoS counts per token")
            plt.xticks(rotation=45)

            # legend
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            plt.legend(*zip(*unique))

            # show/save plot
            plt.show()

    def additional_statistics(self):
        # url count
        command = """
        SELECT count(url)
        FROM GRAPPLING_INSIDER_CONTENT
        """
        self.cursor.execute(command)
        print("url count:", self.cursor.fetchall()[0][0])

        # unique categories
        command = """
        SELECT DISTINCT(category)
        FROM GRAPPLING_INSIDER_CATEGORIES
        """
        self.cursor.execute(command)
        categories = sorted([x[0] for x in hana_handler.cursor.fetchall()])
        print("categories:", categories)
        print("categories count:", len(categories))

    def adj_noun_bigrams(self):
        # url count
        command = """
        SELECT table1.TA_TOKEN, table2.TA_TOKEN
        FROM "$TA_GRAPPLING_INSIDER_INDEX" as table1, "$TA_GRAPPLING_INSIDER_INDEX" as table2
        WHERE table1.URL = table2.URL and table2.TA_COUNTER = table1.TA_COUNTER+1 and table1.TA_TYPE = 'adjective' and table2.TA_TYPE = 'noun'
        """
        self.cursor.execute(command)
        # counter = []
        # adj_nouns = [f"{elem[0]} {elem[1]}" for elem in self.cursor.fetchall()]
        # for adj_noun in set(adj_nouns):
        #     counter.append([adj_noun, adj_nouns.count(adj_noun)])
        # counter = list(reversed(sorted(counter, key=lambda item: item[1])))
        # print(counter)
        counter = {}
        for res in self.cursor:
            res_str = f"{res[0]} {res[1]}"
            if res_str in counter:
                counter[res_str] += 1
            else:
                counter[res_str] = 1
        # sort dict
        counter = dict(reversed(sorted(counter.items(), key=lambda item: item[1])))
        print(counter)

        for k, v in counter.items():
            if v < 75:
                break
            plt.barh(k, v)
        plt.gca().invert_yaxis()
        plt.title("adjective-noun bigrams")
        plt.savefig(os.path.join(self.output_path, "adj_noun_bigrams.png"), bbox_inches="tight")

    def coocc_adj(self):
        # url count
        command = """
                SELECT table1.TA_TOKEN, table2.TA_TOKEN
                FROM "$TA_GRAPPLING_INSIDER_INDEX" as table1, "$TA_GRAPPLING_INSIDER_INDEX" as table2
                WHERE table1.URL = table2.URL and table1.TA_SENTENCE = table2.TA_SENTENCE and table1.TA_COUNTER < table2.TA_COUNTER and table1.TA_TYPE = 'adjective' and table2.TA_TYPE = 'adjective'
                """
        self.cursor.execute(command)
        counter = {}
        for res in self.cursor:
            res_str = f"{res[0]} {res[1]}"
            if res_str in counter:
                counter[res_str] += 1
            else:
                counter[res_str] = 1
        # sort dict
        counter = dict(reversed(sorted(counter.items(), key=lambda item: item[1])))
        print(counter)

        for k, v in counter.items():
            if v < 50:
                break
            plt.barh(k, v)
        plt.gca().invert_yaxis()
        plt.title("cooccurring adjectives (sentence level)")
        plt.savefig(os.path.join(self.output_path, "coocc_adj.png"), bbox_inches="tight")

    def tf_idf_nouns(self):
        # url count
        command = """
                SELECT URL, TA_TOKEN
                FROM "$TA_GRAPPLING_INSIDER_INDEX"
                WHERE TA_TYPE = 'noun'
                ORDER BY URL ASC
                """
        self.cursor.execute(command)
        urls_dict = {}
        nouns_dict = {}
        all_nouns = []
        for res in self.cursor:
            url = res[0]
            noun = res[1]
            all_nouns.append(noun)
            if url in urls_dict:
                urls_dict[url].append(noun)
            else:
                urls_dict[url] = [noun]
            if noun in nouns_dict:
                nouns_dict[noun].append(url)
            else:
                nouns_dict[noun] = [url]
        tf_idf = {}
        url_cnt = len(urls_dict.keys())
        for url, nouns in urls_dict.items():
            for noun in nouns:
                if (url, noun) not in tf_idf:
                    tf = nouns.count(noun)
                    idf = log10(url_cnt/len(nouns_dict[noun]))
                    tf_idf[(url, noun)] = (tf*idf, tf, idf)
        # sort dict
        tf_idf = dict(reversed(sorted(tf_idf.items(), key=lambda item: item[1][0])))
        for i, (k, v) in enumerate(tf_idf.items()):
            if i >= 3:
                break
            print(k, v)

    def get_similar(self, target_url, n=10, mode="cos"):  # modes = ["scalar", "cos"]
        # url count
        command = """
        SELECT URL, TA_TOKEN
        FROM "$TA_GRAPPLING_INSIDER_INDEX"
        WHERE TA_TYPE = 'noun'
        ORDER BY URL ASC
        """
        self.cursor.execute(command)
        # create data
        vecs_dict = {}
        for res in self.cursor:
            url = res[0]
            noun = res[1]
            if url not in vecs_dict:
                vecs_dict[url] = {}
            if noun in vecs_dict[url]:
                vecs_dict[url][noun] += 1
            else:
                vecs_dict[url][noun] = 1
        data = pd.DataFrame(vecs_dict)
        data = data.fillna(0)
        data = data.astype(int)
        data = data.transpose()
        # get vec
        vec = data[data.index == target_url]
        # determine similarity
        if mode == "scalar":
            data["sim"] = data.apply(lambda row: sum(row.values*vec.values[0]), axis=1)
        elif mode == "cos":
            data["sim"] = data.apply(lambda row: sum(row.values*vec.values[0])/sqrt(sum(row.values**2))*sqrt(sum(vec.values[0]**2)), axis=1)
        else:
            print("unknown mode")
            return None
        # sort by similarity
        data = data.sort_values("sim", ascending=False)
        # return top n urls
        return list(data.index)[1:n+1]

    def eval(self, url_list, n=50):
        # url count
        command = """
        SELECT URL, CATEGORY
        FROM GRAPPLING_INSIDER_CATEGORIES
        ORDER BY URL DESC
        """
        self.cursor.execute(command)
        categories_dict = {}
        for res in self.cursor:
            url = res[0]
            cat = res[1]
            if url in categories_dict:
                categories_dict[url].append(cat)
            else:
                categories_dict[url] = [cat]
        categories_dict = dict(reversed(sorted(categories_dict.items(), key=lambda item: len(item[1]))))
        # print(categories_dict)

        # get similar urls to given urls
        for url in url_list:
            target_url_cat = categories_dict[url]
            # print(url, target_url_cat)

            # remove "BJJ News" due to being applied to nearly every document
            target_url_cat.remove("BJJ News")

            # determine truth values for ordered results
            similar_urls = self.get_similar(url, n=n)
            truth_vals = []
            for similar_url in similar_urls:
                similar_url_cat = categories_dict[similar_url]
                # print(similar_url, similar_url_cat)

                # check for overlap
                if not set(target_url_cat).isdisjoint(similar_url_cat):
                    truth_vals.append(True)
                else:
                    truth_vals.append(False)

            # determine precision/recall at i
            precisions = []
            recalls = []
            tp = 0
            fp = 0
            tn = truth_vals.count(False)
            fn = truth_vals.count(True)
            for truth_val in truth_vals:
                if truth_val:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1
                precisions.append(tp/(tp+fp))
                recalls.append(tp/(tp+fn))
            plt.plot(recalls, precisions)

            # determine best precision for recall
            best_precisions = [precisions[0]]
            for i in range(len(recalls[1:])):
                if recalls[i] == recalls[i-1] and precisions[i] < precisions[i-1]:
                    best_precisions.append(best_precisions[-1])
                else:
                    best_precisions.append(precisions[i])
            plt.plot(recalls, best_precisions, color="red")

            # plot settings
            plt.title(url.split("/")[-2].replace("andre-galvao", "a-g"))
            plt.xlabel("recall")
            plt.ylabel("precision")

            plt.savefig(os.path.join(self.output_path, f"{url[len('https://grapplinginsider.com/'):-1]}_precision-recall.png"), bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    hana_handler = HanaHandler()

    # create tables and insert data
    # hana_handler.create_tables()
    # hana_handler.insert_data()

    # create index on text
    # configs = ["LINGANALYSIS_BASIC", "LINGANALYSIS_STEMS", "LINGANALYSIS_FULL"]
    # hana_handler.drop_index()
    # hana_handler.create_index(config=configs[2])

    # lexicon analysis
    # hana_handler.index_analysis()

    # word cloud
    # hana_handler.wordcloud()

    # word ambiguity
    # hana_handler.word_ambiguity()

    # additional statistics
    # hana_handler.additional_statistics()

    # hana_handler.adj_noun_bigrams()
    # hana_handler.coocc_adj()
    # hana_handler.tf_idf_nouns()

    # url_list = ["https://grapplinginsider.com/john-danaher-picks-the-best-open-guard/",
    #             "https://grapplinginsider.com/ryan-hall-still-cant-get-anyone-to-fight-him/",
    #             "https://grapplinginsider.com/5-reasons-why-marcelo-garcia-is-the-greatest-of-all-time/",
    #             "https://grapplinginsider.com/who-has-beaten-more-adcc-medalists-gordon-ryan-or-andre-galvao/"]
    # for url in url_list:
    #     print(url, hana_handler.get_similar(url))
    # hana_handler.eval(url_list)
