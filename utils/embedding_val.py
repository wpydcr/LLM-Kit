
import gradio as gr
from modules.model.emb_auto import AutoEmb
import os
import plotly.graph_objs as go
import random
import numpy as np
import pandas as pd
from openTSNE import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from utils.chinese_text_splitter import ChineseTextSplitter
import json

real_path = os.path.split(os.path.realpath(__file__))[0]



class embedding_visualization_plot():
    def __init__(self):
        self.label = []
        self.content = []
        self.pretext = []
        self.prelabel = []
        self.max_sentences = 20  # 文件提取句子最大数量
        self.real_path = os.path.split(os.path.realpath(__file__))[0]
        self.embed_file_path = os.path.join(self.real_path,"..", "models", "Embedding")
        self.embedding_models = self.get_directories(self.embed_file_path,[])
        test=['公共卫生与预防医学','水产','艺术学','作物学','体育学']
        with open(os.path.join(self.real_path,'dev.json'),'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                if obj["label"] in test:
                    self.pretext.append(obj["content"][:100])
                    self.prelabel.append(obj["label"])

    def get_directories(self, path, unuse):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in unuse]

    def refresh_directories(self):
        '''
            刷新嵌入模型库
        '''
        new_path = os.path.join(self.real_path, "..", "models", "Embedding")
        models = self.get_directories(new_path, [])
        return gr.update(choices=models)

    def check_text_lenth(self,text,label):
        if len(text) > self.max_sentences:
            indices = list(range(len(text)))
            selected_indices = random.sample(indices, self.max_sentences)
            text = [text[i] for i in selected_indices]
            label = [label[i] for i in selected_indices]         
        output = len(text)
        self.label=self.prelabel+label
        self.content=self.pretext+text
        return output

    def upload_data(self, temp_file):
        '''
            上传文件处理, 读取label和内容
        '''
        words_persentence=80        # 定义切分文章每句子长度
        fname=temp_file.name
        suffix = str(fname).split('.')[-1]

        if suffix == "json":
            self.content = []
            self.label = []
            with open(fname,'r',encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    obj = json.loads(line)
                    if len(obj) == 2 and "content" in obj and "label" in obj:
                        self.content.append(obj["content"])
                        self.label.append(obj["label"])
                    elif len(obj) == 1 and "content" in obj:
                        self.content.append(obj["content"])
                        self.label.append("输入文件(★)")
                    else:
                        return "json格式错误,修改json格式: 文件不超过3个key, 第一个是content,第二列是label(可选)"
        elif suffix == "csv":
            self.content = []
            self.label = []
            loader = CSVLoader(fname)
            csv = loader.load()
            for line in csv:
                subtext = line.page_content.split('\n')
                if len(subtext) == 2 and subtext[0].split(': ')[0] == 'content' and subtext[1].split(': ')[0] == 'label':
                    self.content.append(subtext[0].split(': ')[1])
                    self.label.append(subtext[1].split(': ')[1])
                elif len(subtext) == 1 and subtext[0].split(': ')[0] == 'content':
                    self.content.append(subtext[0].split(': ')[1])
                    self.label.append("输入文件(★)")
                else:
                    return "csv格式错误,修改csv格式: 文件不超过3列, 第一列是content,第二列是label(可选)"
        elif suffix == "docx":
            loader = UnstructuredFileLoader(fname, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=words_persentence)
            docs = loader.load_and_split(text_splitter=textsplitter)
            self.content = [i.page_content for i in docs]
            self.label = ["输入文件(★)"] * len(self.content)
        elif suffix == "pdf":
            loader = UnstructuredFileLoader(fname)
            textsplitter = ChineseTextSplitter(pdf=True, sentence_size=words_persentence)
            pdf = loader.load_and_split(textsplitter)
            self.content = [i.page_content for i in pdf]
            self.label = ["输入文件(★)"] * len(self.content)
        elif suffix == "txt":
            loader = TextLoader(fname, encoding='utf-8')
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=words_persentence)
            text = loader.load_and_split(textsplitter)
            self.content = [i.page_content for i in text]
            self.label = ["输入文件(★)"] * len(self.content)
        else:
            return '错误文件输入，要求后缀为 json , csv , docx , txt , pdf'
        if len(self.content) != 0:
            output = self.check_text_lenth(self.content, self.label)
            return f"{suffix}载入成功, 输入语句数" + str(output)
        return f"{suffix}文件载入失败, 请检查文件格式"

    def plot(self, model_names):
        '''
            绘制图片, 返回fig
        '''
        if len(model_names) == 0:
            raise "未选择模型"
        embded_list = []
        for model_name in model_names:
            model = AutoEmb(model_name)
            embeddings = model.embedding.embed_documents(self.content)
            embded_list.append(embeddings)
        fig = go.Figure()
        tsne_plot = self.TSNE_Plot(self.content, embded_list, self.label, model_num=len(model_names), model_name=model_names,
                                   n_components_min=50)
        fig = tsne_plot.tsne_plot()
        return fig

    class TSNE_Plot():
        '''Class TSNE_Plot
        Method
        __init__
            input:
                sentence: List[string]
                    用于展示的语句,应当是一个字符串的list
                embed_list: List[array]
                    用于计算相似度的embedding list,list里每一个元素对应一个模型输出的embedding,应当是np.array(float), 或者tensor(float)等可以直接进行矩阵乘法的array
                label: List
                    用于标注不同颜色的label,应当是一个list,对应每个sentence的label。
                model_num: int
                    使用的embedding模型数量
                model_names:
                    使用模型名字列表。
                n_components_min: int
                    降低到的维度数,默认50维。
        tsne_plot
            input:
                return_fig: [optional] boolean
                    是否返回图像对象,如果为False,将直接绘制图像,如果为True,将返回图像对象,默认为False
            Output:
                若return_fig为True,将返回图像对象
        '''

        def __init__(self, sentence, embed_intup_list, label, model_name, model_num=2, n_components_min: int = 50):
            self.n_components_min = n_components_min
            self.model_name = model_name
            self.test_X_list = []
            self.model_num = model_num
            for j in range(self.model_num):
                self.test_X_list.append(
                    pd.DataFrame({"text": sentence, "embed": [np.array(i) for i in embed_intup_list[j]]}))
            self.test_y = pd.DataFrame({'label': label})
            self.embed_list, self.calinski_harabasz_score_list, self.silhouette_score_list = self.calculate_tsne()

        def generate_colormap(self, n_labels):
            # 创建一个均匀分布的颜色映射
            color_norm = mcolors.Normalize(vmin=0, vmax=len(n_labels) - 1)
            # 使用 plt.cm 中预先定义的colormap,你可以自由选择其他colormap如"hsv", "hot", "cool", "viridis"等
            scalar_map = plt.cm.ScalarMappable(norm=color_norm, cmap='jet')
            colormap = {}
            for label in range(len(n_labels)):
                # 将颜色值转换为十六进制
                color_hex = mcolors.to_hex(scalar_map.to_rgba(label))
                colormap[n_labels[label]] = color_hex
            return colormap

        def show_text(self, text):
            sentence = []
            for i in range(len(text)):
                if len(text[i]) < 75:
                    s = text[i]
                else:
                    s = text[i][:50] + "..." + text[i][-50:]
                sentence.append(s)
            return sentence

        def init_df(self):
            X, Y = np.split(self.embed, 2, axis=1)
            data = {
                "x": X.flatten(),
                "y": Y.flatten(),
            }
            self.df = pd.DataFrame(data)

        def format_data(self):
            sentence2 = self.show_text(self.test_X_list[0]["text"])
            self.df = []
            for i in range(self.model_num):
                X, Y = np.split(self.embed_list[i], 2, axis=1)
                data = {
                    "x": X.flatten(),
                    "y": Y.flatten(),
                    "label": self.test_y["label"],
                    "sentence2": sentence2,
                }
                self.df.append(pd.DataFrame(data))

        def calculate_tsne(self):  # 返回最终的二维嵌入结果,以用于可视化。
            from sklearn.metrics import silhouette_score
            from sklearn.metrics import calinski_harabasz_score
            embedding_train = []
            ch_score = []
            s_score = []
            for i in range(self.model_num):
                embed = np.array(self.test_X_list[i]["embed"].tolist())
                n_components = min(self.n_components_min, len(self.test_X_list[i]))
                pca = PCA(n_components=n_components)
                compact_embedding = pca.fit_transform(embed)
                tsne = TSNE(
                    perplexity=30,
                    metric="cosine",
                    n_jobs=8,
                    random_state=42,
                    verbose=False,
                )
                embedding_train.append(tsne.fit(compact_embedding))
                embedding_train[i] = embedding_train[i].optimize(n_iter=1000, momentum=0.8)
                ch_score.append(calinski_harabasz_score(embedding_train[i], self.test_y))
                s_score.append(silhouette_score(embedding_train[i], self.test_y))
            return embedding_train, ch_score, s_score

        def plot(self, return_fig=False):
            fig = go.Figure()
            label_colors = self.generate_colormap(self.df[0]['label'].unique())
            line_legend_group = "lines"
            set_size = 10  # size 参数控制点大小
            for i in range(self.model_num):
                self.model_name[i] = self.model_name[i] + " (" + str(
                    round(self.calinski_harabasz_score_list[i], 2)) + ")"
            plot_rows = int((self.model_num + 1) / 2)
            fig = go.Figure().set_subplots(rows=plot_rows, cols=2,
                                           shared_xaxes=False,
                                           shared_yaxes=False,
                                           vertical_spacing=0.05,
                                           subplot_titles=self.model_name)
            location = []
            for i in range(plot_rows):
                for j in range(2):
                    location.append([i + 1, j + 1])
            # 为每个类别的点创建散点图
            for i in range(self.model_num):
                for label, color in label_colors.items():
                    mask = self.df[i]["label"] == label
                    fig.add_trace(go.Scatter(x=self.df[i][mask]['x'], y=self.df[i][mask]['y'], mode='markers',
                                             marker=dict(color=color, size=set_size),
                                             hovertext=self.df[i][mask]['sentence2'],  # 鼠标添加文字
                                             showlegend=True if i == 0 else False, legendgroup=line_legend_group,
                                             name=str(label)), row=location[i][0], col=location[i][1]
                                  )
            fig.update_layout(title_text=" Calinski Harabasz (分数越高越好)", height=650 * plot_rows)
            # 取消坐标轴的数字
            fig.update_xaxes(tickvals=[])
            fig.update_yaxes(tickvals=[])
            if not return_fig:
                fig.show()
            else:
                return fig

        def tsne_plot(self, return_fig=True):
            self.format_data()
            if not return_fig:
                self.plot()
            else:
                return self.plot(return_fig=return_fig)
            
