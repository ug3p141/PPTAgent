# DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents

 Tsu-Jui Fu1, William Yang Wang1, Daniel McDuff2, Yale Song2

1 UC Santa Barbara 2 Microsoft Research

{tsu-juifu,william}@cs.ucsb.edu {damcduff, yalesong}@microsoft.com

###### Abstract

Creating presentation materials requires complex multimodal reasoning skills to summarize key concepts and arrange them in a logical and visually pleasing manner. Can machines learn to emulate this laborious process? We present a novel task and approach for document-to-slide generation. Solving this involves document summarization, image and text retrieval, and slide structure to arrange key elements in a form suitable for presentation. We propose a hierarchical sequence-to-sequence approach to tackle our task in an end-to-end manner. Our approach exploits the inherent structures within documents and slides and incorporates paraphrasing and layout prediction modules to generate slides. To help accelerate research in this domain, we release a dataset of about 6K paired documents and slide decks used in our experiments. We show that our approach outperforms strong baselines and produces slides with rich content and aligned imagery.

## 1 Introduction

Creating presentations is often a work of art. It requires skills to abstract complex concepts and conveys them in a concise and visually pleasing manner. Consider the steps involved in creating presentation slides based on a white paper or manuscript: One needs to 1) establish a storyline that will connect with the audience, 2) identify essential sections and components that support the main message, 3) delineate the structure of that content, e.g., the ordering/length of the sections, 4) summarize the content in a concise form, e.g., punchy bullet points, and 5) gather figures that help communicate the message accurately and engagingly.

Can machines emulate this laborious process by _learning_ from the plethora of example manuscripts and slide decks created by human experts? Building such a system poses unique challenges in vision-and-language understanding. Both the input (a manuscript) and output (a slide deck) contain tightly coupled visual and textual elements; thus, it requires multimodal reasoning. Further, there are significant differences in the presentation: compared to manuscripts, slides tend to be more _concise_ (e.g., containing bullet points rather than full sentences), _structured_ (e.g., each slide has a fixed screen real estate and delivers one or few messages), and _visual-centric_ (e.g., figures are first-class citizens, the visual layout plays an important role, etc.).

Existing literature only partially addresses some of the challenges above. Document summarization [1, 10] aims to find a concise text summary of the input, but it does not deal with images/figures and lacks multimodal understanding. Cross-modal retrieval [12, 13] focuses on finding a multimodal embedding space but does not produce summarized outputs. Multimodal summarization [10] deals with both (summarizing documents with text and figures), but it lacks the ability to produce structured output (as in slides). Furthermore, none of the above addresses the challenge of finding an optimal visual layout of each slide. While assessing visual aesthetics have been investigated [11], exiting work focuses on photographic metrics for images that would not translate to slides. These aspects make

Figure 1: We introduce DOC2PPT, a novel task of generating a slide deck from a document. This requires solving several challenges in the vision-and-language domain, e.g., visual-semantic embedding and multimodal summarization. In addition, slides exhibit unique properties such as concise text (bullet points) and stylized layout.

ours a unique task in the vision-and-language literature.

In this paper, we introduce DOC2PPT, a novel task of creating presentation slides from scientific documents. As with no existing benchmark, we collect 5,873 paired scientific documents and associated presentation slide decks (for a total of about 70K pages and 100K slides, respectively). We present a series of automatic data processing steps to extract useful learning signals and introduce new quantitative metrics designed to measure the quality of the generated slides.

To tackle this task, we present a hierarchical recurrent sequence-to-sequence architecture that "reads" the input document and "summarizes" it into a _structured_ slide deck. We exploit the inherent structure within documents and slides by performing inference at the section-level (for documents) and at the slide-level (for slides). To make our model end-to-end trainable, we explicitly encode section/slide embeddings and use them to learn a policy that determines _when to proceed_ to the next section/slide. Further, we learn the policy in a hierarchical manner so that the network decides which actions to take by considering the structural context, e.g., a decision to create a new slide will depend on both the current section and the previous generated content.

To consider the concise nature of text in slides (e.g., bullet points), we incorporate a paraphrasing module that converts document-style full sentences to slide-style phrases/clauses. We show that it drastically improves the quality of the generated textual content for the slides. In addition, we introduce a text-figure matching objective that encourages related text-figure pairs to appear on the same slide. Lastly, we explore both template-based and learning-based layout design and compare them both quantitatively and qualitatively.

Taking a long-term view, our objective is not to take humans completely out of the loop but enhance humans' productivity by generating slides _as drafts_. This would create new opportunities to human-AI collaboration [1], e.g., one could quickly create a slide deck by revising the auto-generated draft and skim them through to digest lots of material. To summarize, our main contributions include: 1) Introducing a novel task, dataset, and evaluation metrics for automatic slide generation; 2) Proposing a hierarchical sequence-to-sequence approach that summarizes a document in a structure output format suitable for slide presentation; 3) Evaluating our approach both quantitatively, using our proposed metrics, and qualitatively based on human evaluation. We hope that our DOC2PPT will advance the state-of-the-art in the vision-and-language domain.

## Related Work

**Vision-and-Language.** Joint modeling of vision-and-language has been studied from different angles. Image/video captioning [23, 24, 25, 26], visual question answering [16, 17, 18], and visually-grounded dialogue generation [14] are all tasks that involve learning relationships between image and text. Despite this large body of work, there remain many tasks that have not been addressed, e.g., multimodal document generation. As argued above, our task brings a new suite of challenges to vision-and-language understanding.

**Document Summarization.** This task has been tackled from two angles: abstractive [15, 26, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

+++ ==WARNING: Truncated because of repetitions==
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 12, 14, 16, 18, 19, 13, 15, 17, 19, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
+++

* An **Object Placer (OP)** decides which object from the current section (sentence or figure) to put on the current slide. It also predicts the location and the size of each object to be placed on the slide;
* A **Paraphraser (PAR)** takes the selected sentence and rewrites it in a concise form before putting it on a slide.

Notation.A document \(\mathcal{D}\) is organized into sections \(\mathcal{S}=\{S_{i}\}_{i\in N^{in}_{S}}\) and figures \(\mathcal{F}=\{F^{in}_{q}\}_{q\in M^{in}_{F}}\). Each section \(S_{i}\) contains sentences \(\mathcal{T}^{in}_{i}=\{\mathcal{T}^{in}_{i,k}\}_{k\in N^{in}_{i}}\), and each figure \(F_{q}=\{I_{q},C_{q}\}\) contains an image \(\hat{I}_{q}\) and a caption \(C_{q}\). We do not assign figures to any particular section because multiple sections can reference the same figure. A slide deck \(\mathcal{O}=\{O_{j}\}_{j\in N^{out}_{O}}\) contains a number of slides, each containing sentences \(\mathcal{T}^{out}_{j}=\{T^{out}_{j,k}\}_{k\in N^{out}_{j}}\) and figures \(\mathcal{F}^{out}_{j}=\{F^{out}_{j,k}\}_{k\in M^{out}_{F}}\). We encode the position and the size of each object on a slide in a bounding box format using an auxiliary layout variable \(L_{j,k}\), which includes four real-valued numbers \(\{l^{x},l^{y},l^{w},l^{h}\}\) encoding the x-y offsets (top-left corner), the width and height of a bounding box.

Model Document Reader (DR).We extract sentence and figure embeddings from an input document and project them to a shared embedding space so that the OP treats both textual and visual elements as an object coming from a joint multimodal distribution. For each section \(S_{i}\), we use RoBERTa Liu et al. (2019) to encode each of the sentences \(T^{in}_{i,k}\), and then use a bidirectional GRU Chung et al. (2014) to extract contextualized sentence embeddings \(X^{in}_{i,k}\):

\[\begin{split} B^{in}_{i,k}&=\text{RoBERTa}(T^{in}_{ i,k}),\\ X^{in}_{i,k}&=\text{Bi-GRU}(B^{in}_{i,0},...,B^{in }_{i,N^{in}_{i}-1})_{k},\end{split} \tag{1}\]

Similarly, for each figure \(F^{in}_{q}=\{I^{in}_{q},C^{in}_{q}\}\), we apply ResNet-152 He et al. (2016) to extract the image embedding of \(I^{in}_{q}\) and RoBERTa for the caption embedding of \(C^{in}_{q}\). We then concatenate them as the figure embedding \(V^{in}_{q}\):

\[V^{in}_{q}=[\text{ResNet}(F^{in}_{q}),\text{RoBERTa}(C^{in}_{q})]. \tag{2}\]

Next, we project \(X^{in}_{i,k}\) and \(V^{in}_{q}\) to a shared embedding using a two-layer multilayer perceptron (MLP) and combine \(E^{txt}_{i}\) and \(E^{fig}\) as the section embedding \(E^{sec}_{i}\) of \(S_{i}\):

\[\begin{split} E^{txt}_{i,k}=\text{MLP}^{txt}(X^{in}_{i,k}),& \quad E^{fig}_{q}=\text{MLP}^{fig}(V^{in}_{q}),\\ E^{sec}_{i}=\{E^{txt}_{i,k},E^{fig}_{q}\}_{k\in N^{in}_{i},q \in M^{in}_{F}}\end{split} \tag{3}\]

We include all figures \(\mathcal{F}\) in _each_ section embedding \(E^{sec}_{i}\) because each section can reference any of the figures.

Progress Tracker (PT).We define the PT as a state machine operating in a hierarchically-structured space with sections ([SEC]), slides ([SLIDE]), and objects ([OBJ]). This is to reflect the structure of documents and slides, i.e., each section of a document can have multiple corresponding slides, and each slide can contain multiple objects.

The PT maintains pointers to the current section \(i\) and the current slide \(j\), and learns a policy to proceed to the next section/slide as it generates slides. For simplicity, we initialize \(i=j=0\), i.e., the output slides will follow the natural order of sections in an input document. We construct PT as a three-layer hierarchical RNN with (\(\texttt{PT}^{sec},\texttt{PT}^{slide},\texttt{PT}^{obj}\)), where each RNN encodes the latent space for each level in a section-slide-object hierarchy. This is a natural choice to encode our prior knowledge about the hierarchical structure.

First, \(\texttt{PT}^{sec}\) takes as input the head-tail contextualized sentence embeddings from the DR, which encodes the overall information of the current section \(S_{i}\). We use GRU for \(\texttt{PT}^{sec}\) and initialize \(h^{sec}_{0}\) to the contextualized sentence embeddings of the first section, i.e., \(h^{sec}_{0}=[X^{in}_{0,1},X^{in}_{0,N^{in}_{0}-1}]\):

\[h^{sec}_{i}=\texttt{PT}^{sec}(h^{sec}_{i-1},[X^{in}_{i,1},X^{in}_{i,N^{in}_{i} }]), \tag{4}\]

Based on the section state \(h^{sec}_{i}\), \(\texttt{PT}^{slide}\) models the section-to-slide relationships:

\[a^{sec}_{j},h^{slide}_{j}=\texttt{PT}^{slide}(a^{sec}_{j-1},h^{slide}_{j-1},E^ {sec}_{i}), \tag{5}\]

where \(h^{slide}_{0}=h^{sec}_{i}\), \(E^{sec}_{i}\) is the section embedding (Eq. 3), and \(a^{sec}_{j}\) is a binary action variable that tracks the section pointer, i.e, it decides if the model should generate a new slide for the current section \(S_{i}\) or proceed to the next section \(S_{i+1}\). We implement \(\texttt{PT}^{slide}\) as a GRU and a two-layer MLP with a binary decision head that learns a policy \(\phi\) to predict \(a^{sec}_{j}=\{\{\texttt{NEW\_SLIDE}\},\{\texttt{END\_SEC}\}\}\):

\[\begin{split} a^{sec}_{j}=\text{MLP}^{slide}_{\phi}([h^{slide}_{j}, \sum_{r}\alpha^{slide}_{j,r}E^{sec}_{i,r}]),\\ \alpha^{slide}_{j}=\text{softmax}(h^{slide}_{j}W(E^{sec}_{i})^{ \intercal}).\end{split} \tag{6}\]

\(\alpha^{slide}_{j}\in\mathbb{R}^{N^{in}_{i}+M^{in}}\) is an attention map over \(E^{sec}_{i}\) that computes the bilinear compatibility between \(h^{slide}_{j}\) and \(E^{sec}_{i}\).

Figure 2: An overview of our architecture. It consists of modules (DR, PT, OP, PAR) that read a document and generate a slide deck in a hierarchically structured manner.

Finally, the object \(\texttt{PT}^{obj}\) tracks which objects to put on the current slide \(O_{j}\) based on the slide state \(h_{j}^{slide}\):

\[\begin{split} a_{k}^{slide},h_{k}^{obj}&=\texttt{PT} ^{obj}(a_{k-1}^{slide},h_{k-1}^{obj},E_{i}^{sec}),\\ a_{k}^{slide}&=\text{MLP}_{\psi}^{obj}([h_{k}^{ obj},\sum_{r}\alpha_{k,r}^{obj}E_{i,r}^{sec}]),\\ \alpha_{k}^{obj}&=\text{softmax}(h_{k}^{obj}W(E_{i }^{sec})^{\intercal}).\end{split} \tag{7}\]

Similarly, \(a_{k}^{slide}=\{\,\texttt{[NEW\_OBJ]},\,\texttt{[END\_SLIDE]}\}\) is a binary action variable that decides whether to put a new object for the current slide or proceed to the next. We again set \(h_{0}^{obj}=h_{j}^{slide}\) and use a GRU and a two-layer \(\text{MLP}_{\psi}\) to implement \(\texttt{PT}^{obj}\), together with an attention matrix \(W\) between \(h_{k}^{obj}\) and \(E_{i}^{sec}\). Note that each of the three PTs have an independent set of weights to ensure that they model distinctive dynamics in the section-slide-object structure.

Object Placer (OP).When \(\texttt{PT}^{obj}\) takes an action \(a_{k}^{slide}=\texttt{[NEW\_OBJ]}\), the OP selects an object from the current section \(S_{i}\) and predicts the location on the current slide \(O_{j}\) in which to place it. For this, we use the attention score \(\alpha_{k}^{obj}\) to choose an object (sentence or figure) that has the maximum compatibility score with the current object state \(h_{k}^{obj}\), i.e., \(\arg\max_{r}\alpha_{k}^{obj}\). We then employ a two-layer MLP to predict the layout variable for the chosen object:

\[\{l_{k}^{x},l_{k}^{y},l_{k}^{w},l_{k}^{h}\}=\text{MLP}^{layout}([h_{k}^{obj}, \sum_{r}\alpha_{k,r}^{obj}E_{i,r}^{sec}]). \tag{8}\]

Note that the distinctive style of presentation slides requires special treatment of the objects. If an object is a figure, we take only the image part and resize it to fit the bounding box region while maintaining the original aspect ratio. If an object is a sentence, we first paraphrase it into a concise form and also adjust the font size to fit inside.

Paraphraser (PAR).We paraphrase sentences before placing them on slides. This step is crucial because without it the text would be too verbose for a slide presentation.2 We implement the PAR as Seq2Seq [1] with the copy mechanism [10]:

Footnote 2: In our dataset, sentences in the documents have an average of 17.3 words, while sentences in slides have 11.6 words; the difference is statistically significant (\(p=0.0031\)).

\[\{w_{0},...,w_{l-1}\}=\text{PAR}(T_{j,k}^{out},h_{k}^{obj}), \tag{9}\]

where \(T_{j,k}^{out}\) is a sentence chosen by OP. We condition PAR on the object state \(h_{k}^{obj}\) to provide contextual information and demonstrate this importance in the supplementary.

### Training

We design a learning objective that captures both the structural similarity and the content similarity between the ground-truth slides and the generated slides.

Structural similarity.The series of actions \(a_{j}^{sec}\) and \(a_{k}^{slide}\) determines the _structure_ of output slides. To encourage our model to generate slide decks with a similar structure as the ground-truth, we adopt the the cross-entropy loss (CE) and define our structural similarity loss as:

\[\mathcal{L}_{structure}=\sum\nolimits_{j}\text{CE}(a_{j}^{sec})+\sum\nolimits_ {k}\text{CE}(a_{k}^{slide}). \tag{10}\]

Content Similarity.We formulate our content similarity loss to capture various aspects of slide generation quality, measuring whether the model 1) selected important sentences and figures from the input document, 2) adequately phrased sentences in the presentation style (e.g., shorter sentences), 3) placed sentences and figures to the right locations on a slide, and 4) put sentences and figures on a slide that are relevant to each other. We define our content similarity loss to measure each of the four aspects described above:

\[\begin{split}\mathcal{L}_{content}=\sum\nolimits_{k}\text{CE}( \alpha_{k}^{obj})+\sum\nolimits_{l}\text{CE}(w_{l})+\\ \sum\nolimits_{u,v}\text{CE}(\delta([E_{u}^{txt},E_{v}^{fig}]))+ \sum\nolimits_{k}\text{MSE}(L_{k}).\end{split} \tag{11}\]

Selection loss (\(\alpha_{k}^{obj}\)).The first term checks whether it selected the "correct" objects that also appear in the ground truth. This term is slide-insensitive, i.e., the correct/incorrect inclusion is not affected by which specific slide it appears in.

Paraphrasing loss (\(w_{l}\)).The second term measures the quality of paraphrased sentences by comparing the output sentence and the ground-truth sentence word-by-word.

Text-Figure matching loss (\(\delta([E_{u}^{txt},E_{v}^{fig}])\)).The third term measures the relevance of text and figures appearing in the same slide. We follow the literature on visual-semantic embedding [13, 14, 15] and learn an additional multimodal projection head \(\delta([E_{u}^{txt},E_{v}^{fig}])\) with a sigmoid activation that outputs a relevance score in \([0,1]\). For positive training pairs, we sample text-figure pairs from a) ground-truth slides and b) paragraph-figure pairs where the figure is mentioned in that paragraph. We randomly construct negative pairs.

Layout loss (\(L_{k}\)).The last term measures the quality of slide layout by regressing the predicted bounding box to the ground-truth. While there exist several solutions to bounding box regression [10, 11], we opted for the simple mean squared error (MSE) computed directly over the layout variable \(L_{k}=\{l_{k}^{x},l_{k}^{y},l_{k}^{w},l_{k}^{h}\}\).

The Final Loss.We define our final learning objective as:

\[\mathcal{L}_{DOC2PPT}=\mathcal{L}_{structure}+\gamma\mathcal{L}_{content} \tag{12}\]

where \(\gamma\) controls the relative importance between structural and content similarity; we set \(\gamma=1\) in our experiments.

To train our model, we follow the standard teacher-forcing approach [12] for the sequential prediction and provide the ground-truth results for the past prediction steps, e.g., the next actions \(a_{j}^{sec}\) and \(a_{k}^{slide}\) are based on the ground-truth actions \(\tilde{a}_{j-1}^{sec}\) and \(\tilde{a}_{k-1}^{slide}\), the next object \(\alpha_{k}^{obj}\) is selected based on the ground-truth object \(\tilde{\alpha}_{k-1}^{obj}\), etc.

### Inference

The inference procedures during training and test times largely follow the same process, with one exception: At test time, we utilize the multimodal projection head \(\delta(\cdot)\) to act as a post-processing tool. That is, once our model generates a slide deck, we remove figures that have relevance scores lower than a threshold \(\theta^{R}\) and add figures with scores higher than a threshold \(\theta^{A}\). We tune the two hyper-parameters \(\theta^{R}\) and \(\theta^{A}\) via cross-validation (we set \(\theta^{R}=0.8\), \(\theta^{A}=0.9\)).



## Dataset

We collect pairs of documents and the corresponding slide decks from academic proceedings, focusing on three research communities: computer vision (CVPR, ECCV, BMVC), natural language processing (ACL, NAACL, EMNLP), and machine learning (ICML, NeurIPS, ICLR). Table 1 reports the descriptive statistics of our dataset.

For the training and validation set, we automatically extract text and figures from documents and slides and perform matching to create document-to-slide correspondences. To ensure that our test set is clean and reliable, we use Amazon Mechanical Turk (AMT) and have humans perform image extraction and matching for the entire test set. We provide an overview of our extraction and matching processes; including details of data collection and extraction/matching processes with reliability analyses in the supplementary.

**Text and Figure Extraction.** For each document \(\mathcal{D}\), we extract sections \(\mathcal{S}\) and sentences \(\mathcal{T}^{in}\) using ScienceParse [10] and figures \(\mathcal{F}^{in}\) using PDFFigures [11]. For each slide deck \(\mathcal{O}\), we extract sentences \(\mathcal{T}^{out}\) using Azure OCR [14] and figures \(\mathcal{F}^{out}\) using the border following technique [22, 10].

**Slide Stemming.** Many slides are presented with animations, and this makes \(\mathcal{O}\) contain some successive slides that have similar content minus one element on the preceding slide. For simplicity we consider these near-duplicate slides as redundant and remove them by comparing text and image contents of successive slides: if \(O_{j+1}\) covers more than 80% of the content of \(O_{j}\) (per text/visual embeddings) we discard it and keep \(O_{j+1}\) as it is deemed more complete.

**Slide-Section Matching.** We match slides in a deck to the sections in the corresponding document so that a slide deck is represented as a set of non-overlapping slide groups each with a matching section in the document. To this end, we use RoBERTa [10] to extract embeddings of the text content in each slide and the paragraphs in each section of the document. We assume that a slide deck follows the section order of the corresponding document, and use dynamic programming to find slide-to-section matching based on the cosine similarity between text embeddings.

**Sentence Matching.** We match sentences from slides to the corresponding document. We again use RoBERTa to extract embeddings of each sentence in slides and documents, and search for the matching sentence based on the cosine similarity. We limit the search space only within the corresponding sections using the slide-section matching result.

**Figure Matching.** Lastly, we match figures from slides to those in the corresponding document. We use MobileNet [12] to extract visual embeddings of all \(I^{in}\) and \(I^{out}\) and match them based on the highest cosine similarity. Note that some figures in slides do not appear in the corresponding document (and hence no match). For simplicity, we discard \(F^{out}\) if its highest visual embedding similarity is lower than a threshold \(\theta^{I}=0.8\).

## Experiments

DOC2PPT is a new task with no established evaluation metrics and baselines. We propose automatic metrics specifically designed for evaluating slide generation methods. We carefully ablate various components of our approach and evaluate them on our proposed metrics. We also perform human evaluation to assess the generation quality.

### Evaluation Metrics

**Slide-Level ROUGE (ROUGE-SL).** To measure the quality of text in the generated slides, we adapt the widely-used ROUGE score [10]. Note that ROUGE does not account for the text length in the output, which is problematic for presentation slides (e.g., text in slides are usually shorter). Intuitively, the number of slides in a deck is a good proxy for the overall text length. If too short, too much text will be put on the same slide, making it difficult to read; conversely, if a deck has too many slides, each slide can convey only little information while making the whole presentation lengthy. Therefore, we propose the slide-level ROUGE:

\[\text{ROUGE-SL}=\text{ROUGE-L}\times e^{\frac{|Q-Q|}{Q}}, \tag{13}\]

where \(Q\) and \(\bar{Q}\) are the number of slides in the generated and the ground-truth slide decks, respectively.

**Longest Common Figure Subsequence (LC-FS).** We measure the quality of figures in the output slides by considering both the correctness (whether the figures from the ground-truth deck are included) and the order (whether all the figures are ordered logically - i.e, in a similar manner to the ground-truth deck). To this end, we use the Longest Common Subsequence (LCS) to compare the list of figures in the output \(\{I^{out}_{0},I^{out}_{1},...\}\) to the ground-truth \(\{\tilde{I}^{out}_{0},\tilde{I}^{out}_{1},...\}\) and report precision/recall/F1.

**Text-Figure Relevance (TFR).** A good slide deck should put text with relevant figures to make the presentation informative and attractive. We consider text and figures simultaneously and measure their relevance by a modified ROUGE:

\[\text{TFR}=\frac{1}{M^{in}_{F}}\sum_{i=0}^{M^{|\text{F}^{in}-1}_{i=0}}\text{ ROUGE-L}(P_{i},\tilde{P}_{i}), \tag{14}\]

\begin{table}
\begin{tabular}{l r r r r r r} \hline  & \multicolumn{1}{c}{**Document - Slide**} & \multicolumn{3}{c}{**Documents**} & \multicolumn{3}{c}{**Slides**} \\ \cline{2-7}  & Train / Val / Test & \#Sections & \#Sentences & \#Figures & \#Slides & \#Sentences & \#Figures \\ \hline CV & 2,073 / 265 / 262 & 15,588 (6.0) & 721,048 (46.3) & 24,998 (9.6) & 37,969 (14.6) & 124,924 (8.0) & 4,290 (1.7) \\ NLP & 741 / 93 / 97 & 7,743 (8.3) & 234,764 (30.3) & 8,114 (8.7) & 19,333 (20.8) & 63,162 (8.2) & 3,956 (4.2) \\ ML & 1,872 / 234 / 236 & 17,735 (7.6) & 801,754 (45.2) & 15,687 (6.7) & 41,544 (17.7) & 142,698 (8.0) & 6,187 (2.6) \\ \hline Total & 4,686 / 592 / 595 & 41,066 (6.99) & 1,757,566 (42.8) & 48,799 (8.3) & 98,856 (16.8) & 330,784 (8.1) & 14,433 (2.5) \\ \hline \end{tabular}
\end{table}
Table 1: Descriptive statistics of our dataset. We report both the total count and the average number (in parenthesis).

where \(P_{i}\) and \(\bar{P}_{i}\) are sentences from generated and ground-truth slides that contain \(I_{i}^{in}\), respectively.

Mean Intersection over Union (mIoU).A good design layout makes it easy to consume information presented in slides. To evaluate the layout quality, we adapt the mean intersection over union (mIoU) [11] by incorporating the LCS idea with the ground-truth \(\tilde{\mathcal{O}}\):

\[\text{mIoU}(\mathcal{O},\tilde{\mathcal{O}})=\frac{1}{N_{O}^{out}}\sum\nolimits_ {i=0}^{N_{O}^{out}-1}\text{IoU}(O_{i},\tilde{O}_{J_{i}}) \tag{15}\]

where \(\text{IoU}(O_{i},\tilde{O}_{j})\) computes the IoU between a set of predicted bounding boxes from slide \(i\) and a set of ground-truth bounding boxes from slide and \(J_{i}\). To account for a potential structural mismatch (with missing/extra slides), we find the \(J=\{j_{0},j_{1},...,j_{N_{O}^{out}-1}\}\) that achieves the maximum mIoU between \(\mathcal{O}\) and \(\tilde{\mathcal{O}}\) in an increasing order.

Implementation Detail

For the DR, we use a Bi-GRU with 1,024 hidden units and set the MLPs to output 1,024-dimensional embeddings. Each layer of the PT is based on a 256-unit GRU. The PAR is designed as Seq2Seq [1] with 512-unit GRU. All the MLPs are two-layer fully-connected networks. We train our network end-to-end using ADAM [10] withlearning rate 3e-4.

Results and Discussions

Is the Hierarchical Modeling Effective?We define a "flattened" version of our PT (flat-PT) by replacing the hierarchical RNN with a vanilla RNN that learns a single shared latent space to model the section-slide-object structure. The flat-PT contains a single GRU and a two-layer MLP with a ternary decision head that learns to predict an action \(a_{t}=\{\)[NEW_SECTION], [NEW_SLIDE], [NEW_OBJ] \(\}\). For a fair comparison, we increase the number of hidden units in the baseline GRU to 512 (ours is 256) so the model capacities are roughly the same between the two.

First, we compare the structural similarity between the generated and the ground-truth slide decks. For this, we build a list of tokens indicating a section-slide-object structure (e.g., [SEC], [SLIDE], [OBJ],..., [SLIDE],...) and compare the lists using the LCS. Our hierarchical approach achieves 64.15% vs. the flat-PT 51.72%, suggesting that ours was able to learn the structure better than baseline.

Table 2 (a) and (b) compare the two models on the four metrics. The results show that ours outperforms flat-PT across all metrics. The flat-PT achieves slightly better performance on ROUGE-SL without the slide-length term (w/o SL), which is the same as ROUGE-L. This suggests that ours generates a slide structure more similar to the ground-truth.

A Deeper Look into the Content Similarity Loss.We ablate different terms in the content similarity loss (Eq. 11) to understand their individual effectiveness in Table 2.

PAR.The paraphrasing loss improves text quality in slides; see the ROUGE-SL scores of (b) vs. (c), and (d) vs. (e). It also improves the TFR metric because any improvement in text quality will benefit text-figure relevance.

TIM.The text-figure matching loss improves the figure quality; see (b) vs. (d) and (c) vs. (e). It particularly improves LC-FS precision with a moderate drop in recall rate, indicating the model added more correct figures. TIM also improves ROUGE-SL because it helps constrain the multimodal embedding space, resulting in better selection of text.

Figure Post-Processing.At test time, we leverage the multimodal projection head \(\delta(\cdot)\) as a post-processing module to add missing figures and/or remove unnecessary ones. Table 2 (f) shows this post-processing further improves the two image-related metrics, LC-FS and TFR. For simplicity, we add figures following equally fitting in template-based design instead of using OP to predict its location.

Layout Prediction vs. Template.The OP predicts the layout to decide where and how to put the extracted objects. We compare this with a template-based approach, which selects the current section title as the slide title and puts sentences

\begin{table}
\begin{tabular}{c c c c c||c c|c c c|c} \hline \hline  & \multicolumn{3}{c||}{Ablation Settings} & \multicolumn{3}{c|}{ROUGE-SL} & \multicolumn{3}{c|}{LC-FS} & \multirow{2}{*}{TFR} & \multicolumn{2}{c}{mIoU} \\  & Hrch-PT & PAR & TIM & Post Proc. & Ours & w/o SL & Prec. & Rec. & F1 & \multirow{2}{*}{TFR} & \multirow{2}{*}{(Layout / Template)} \\ \hline (a) & ✗ & ✗ & ✗ & ✗ & 24.35 & 29.77 & 25.54 & 14.85 & 18.78 & 5.61 & 43.34 / 38.15 \\ (b) & ✓ & ✗ & ✗ & ✗ & 24.93 & 29.68 & 17.48 & 26.26 & 20.99 & 8.58 & 49.16 / 40.94 \\ (c) & ✓ & ✓ & ✗ & ✗ & 27.19 & 32.27 & 17.48 & 26.26 & 20.99 & 9.23 & 49.16 / 40.94 \\ (d) & ✓ & ✗ & ✓ & ✗ & 26.52 & 30.99 & 23.47 & 25.31 & 24.36 & 10.09 & **50.82** / 42.96 \\ (e) & ✓ & ✓ & ✓ & ✗ & **29.40** & **34.27** & 23.47 & 25.31 & 24.36 & 11.82 & **50.82** / 42.96 \\ \hline (f) & ✓ & ✓ & ✓ & ✓ & **29.40** & **34.27** & **26.36** & **38.39** & **31.26** & **17.49** & - / 46.73 \\ \hline \hline \end{tabular}
\end{table}
Table 2: Overall result of different ablation settings under automatic evaluation metrics ROUGE-SL, LC-FS, TFR, and mIoU.

\begin{table}
\begin{tabular}{c c c c c} \hline \hline Train \(\downarrow\) / Test \(\rightarrow\) & CV & NLP & ML & All \\ \hline CV & **31.2** / **32.1** / **19.7** & 24.1 / 21.5 / 5.6 & 24.0 / 25.6 / 11.2 & 24.7 / 29.2 / 15.8 \\ NLP & 28.8 / 30.0 / 13.4 & **34.7** / **30.7** / **11.8** & 29.2 / 32.7 / 15.3 & 28.9 / 30.9 / 13.6 \\ ML & 21.1 / 29.2 / 11.6 & 21.1 / 26.6 / 6.6 & **32.1** / **36.8** / **22.8** & 24.9 / **31.4** / 14.4 \\ All & 29.2 / 31.2 / 18.6 & 30.0 / 28.8 / 9.7 & 29.4 / 32.9 / 20.6 & **29.4** / 31.3 / **17.5** \\ \hline \hline \end{tabular}
\end{table}
Table 3: Topic-aware evaluation results (ROUGE-SL / LC-F1 / TFR) when trained and tested on data from different topics.

and figures in the body line-by-line. For those extracted figures, they will equally fit (with the same width) in the remaining space under the main content. The result shows that the predicted-based layout, which directly learns from the layout loss, can bring out higher mIoU with the groundtruth. And in the aspect of the visualization, the template-based design can make the generated slide deck more consistent.

Topic-Aware Evaluation.We evaluate performance in a topic-dependent and independent fashion. To do this, we train and test our model on data from each of the three research communities (CV, NLP, and ML). Table 3 shows that models trained and tested within each topic performs the best (not surprisingly), and that models trained on data from all topics achieves the second best performance, showing generalization to different topic areas. Training on NLP data, despite being the smallest among the three, seems to generalize well to other topics on the text metric, achieving the second best on ROUGE-SL (28.9). Training on CV data provides the second highest performance on the text-figure metric TFR (15.8), and training on ML achieves the highest figure extraction performance (LC-FS F1 of 31.4).

Human Evaluation.We conduct a user study to assess the perceived quality of generates slides. To make the task easy to complete, we sample 200 sections from 50 documents and create 600 pairs of ground-truth and generate slides. We prepare four slide decks per document: the ground-truth deck, and the ones generated by the flat PT (Table 2 (a)), by ours without PAR and TIM (b), and by our final model (f).

We recruited three AMT Master Workers for each task (HIT). The workers were shown the slides from the ground-truth deck (DECK A) and one of the methods (DECK B). The workers were then asked to answer three questions: Q1. Looking only at the TEXT on the slides, how similar is the content on the slides in DECK A to the content on the slides in DECK B?; Q2. How well do the figure(s)/tables(s) in DECK A match the text or figures/tables in DECK B?; Q3. How well do the figure(s)/table(s) in DECK A match the TEXT in DECK B? The responses were all on a scale of 1 (not similar at all) to 7 (very similar). Fig. 4 shows the average scores for each method. The average rating for our approach was significantly greater for all three questions compared to the other two methods. There was no significant difference between the ratings for the other two methods.

Qualitative Results.Fig. 3 illustrates two qualitative examples of the slide deck generated by our model with the template-based layout generation approach. With the post-processing, TIM can add the related figure into the slide and make it more informative. PAR helps create a better presentation by paraphrasing the sentences into bullet point form.

## Conclusion

We present a novel task and approach for generating slides from documents. This is a challenging multimodal task that involves understanding and summarizing documents containing text and figures and structuring it into a presentation form. We release a large set of 5,873 paired documents and slide decks, and provide evaluation metrics with our results. We hope our work will help advance the state-of-the-art in vision-and-language understanding.

Figure 4: The average scores for how closely the generated slides match the text and figures in the ground-truth slides. And how well the generated text matches the figures in the ground-truth slides. Error bars reflect standard error. Significance tests: two-sample t-test (\(p<\)0.05.)

Figure 3: Qualitative examples of the generated slide deck from our model (Paper source: top [16] and bottom [2]). We provide more results on our project webpage: [https://doc2ppt.github.io](https://doc2ppt.github.io)



## References

* A. Agrawal, J. Lu, S. Antol, M. Mitchell, C. L. Zitnick, D. Batra, and D. Parikh (2015)TGIF-QA: toward spatio-temporal reasoning in visual question answering. In ICCV, Cited by: SS1, SS2.
* S. Amershi, D. Weld, M. Vorvoreanu, A. Fourney, B. Nushi, P. Collisson, J. Suh, S. Iqbal, P. Bennett, K. Inken, J. Teevan, R. Kikin-Gil, and E. Horvitz (Eds.) Guidelines for Human-AI Interaction. In CHI, Cited by: SS1, SS2.
* P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang (2018)Bottom-up and top-down attention for image captioning and visual question answering. In CVPR, Cited by: SS1, SS2.
* D. Bahdanau, K. Cho, and Y. Bengio (2015)Neural machine translation by jointly learning to align and translate. In ICLR, Cited by: SS1, SS2.
* F. Barrios, F. Lopez, L. Argerich, and R. Wachenchauzer (2015)Variations of the similarity function of textRank for automated summarization. In ASAI, Cited by: SS1, SS2.
* A. Celikyilmaz, A. Bosselt, X. He, and Y. Choi (2018)Deep communicating agents for abstractive summarization. In NAACL, Cited by: SS1, SS2.
* L. Chen, R. G. Lopes, B. Cheng, M. D. Collins, E. D. Cubuk, B. Zoph, H. Adam, and J. Shlens (2020)Semi-supervised learning in video sequences for urban scene segmentation. In ECCV, Cited by: SS1, SS2.
* X. Chen, S. Gao, C. Tao, Y. Song, D. Zhao, and R. Yan (2018)Iterative document representation learning towards summarization with polishing. In EMNLP, Cited by: SS1, SS2.
* J. Cheng and M. Lapata (2016)Neural summarization by extracting sentences and words. In ACL, Cited by: SS1, SS2.
* J. Cho, M. Seo, and H. Hajishirzi (2019)Mixture content selection for diverse sequence generation. In EMNLP-IJCNLP, Cited by: SS1, SS2.
* S. Chopra, M. Auli, and A. M. Rush (2016)Abstractive sentence summarization with attentive recurrent neural networks. In NAACL, Cited by: SS1, SS2.
* J. Chung, C. Gulcehre, K. Cho, and Y. Bengio (2014)Empirical evaluation of gated recurrent neural networks on sequence modeling. In NeurIPS WS, Cited by: SS1, SS2.
* C. Clark and S. Divvala (2016)PDFFigures 2.0: mining figures from research papers. In JCDL, Cited by: SS1, SS2.
* A. Das, S. Kottur, K. Gupta, A. Singh, D. Yadav, J. M. F. Moura, D. Parikh, and D. Batra (2017)Visual dialog. In CVPR, Cited by: SS1, SS2.
* J. B. Diederik P. Kingma and J. B. Ba (2014)Adam: a method for stochastic optimization. In ICLR, Cited by: SS1, SS2.
* L. Dong, N. Yang, W. Wang, F. Wei, X. Liu, Y. Wang, J. Gao, M. Zhou, and H. Hon (2019)Unified language model pre-training for natural language understanding and generation. In NeurIPS, Cited by: SS1, SS2.
* A. Elkiss, S. Shen, A. Fader, G. Erkan, D. States, and D. Radkov (2008)Blind Men and elephants: what do citation summaries tell us about a research article?. In JA-SIST, Cited by: SS1, SS2.
* M. Everingham, L. V. Gool, C. K. I. Williams, J. Winn, and A. Zisserman (2010)The pascal visual object classes (voc) challenge. In IJCV, Cited by: SS1, SS2.
* F. Faghri, D. J. Fleet, J. R. Kiros, and S. Fidler (2018)VSE++: improving visual-semantic embeddings with hard negatives. In BMVC, Cited by: SS1, SS2.
* A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. Ranzato, and T. Mikolov (2013)DeViSE: a deep visual-semantic embedding model. In NeurIPS, Cited by: SS1, SS2.
* J. Gu, J. Cai, S. Joty, L. Niu, and G. Wang (2018)Look, maging and match: improving textual-visual cross-modal retrieval with generative models. In CVPR, Cited by: SS1, SS2.
* J. Gu, Z. Lu, H. Li, V. V. O. K. V. V. Li, and L. Li (2016)Incorporating copying mechanism in sequence-to-sequence learning. In ACL, Cited by: SS1, SS2.
* K. He, X. Zhang, S. Ren, and J. Sun (2015)Spatial pyramid pooling in deep convolutional networks for visual recognition. In TPAMI, Cited by: SS1, SS2.
* K. He, X. Zhang, S. Ren, and J. Sun (2016)Deep residual learning for image recognition. In CVPR, Cited by: SS1, SS2.
* A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam (2017)MobileNets: efficient convolutional neural networks for mobile vision applications. In CVPR, Cited by: SS1, SS2.
* Y. Hu and X. Wan (2013)PPSGEN: learning to generate presentation slides for academic papers. In IJCAI, Cited by: SS1, SS2.
* Y. Huang, Q. Wu, and L. Wang (2018)Learning semantic concepts and order for image and sentence matching. In CVPR, Cited by: SS1, SS2.
* P. Izmailov, P. Kirichenko, M. Finzi, and A. G. Wilson (2020)Semi-supervised learning with normalizing flows. In ICML, Cited by: SS1, SS2.
* K. Jaidka, M. Kumar, C. karan, and S. R. amd Min-Yen Kan (2016)Overview of the cl-sciSumm 2016 shared task. In BIRNDL, Cited by: SS1, SS2.
* Y. Jang, Y. Song, Y. Yu, Y. Kim, and G. Kim (2017)VQA: visual question answering. In CVPR, Cited by: SS1, SS2.
* A. Karpathy and L. Fei-Fei (2014)Deep visual-semantic alignments for generating image descriptions. In CVPR, Cited by: SS1, SS2.
* R. Kiros, R. Salakhutdinov, and R. S. Zemel (2014)Unifying visual-semantic embeddings with multimodal neural language models. In NeurIPS WS, Cited by: SS1, SS2.
* G. Lev, M. Shmueli-Scheuer, J. Herzig, A. Jerbi, and D. Konopnicki (2019)TalkSumm: a dataset and scalable annotation method for scientific paper summarization based on conference talks. In ACL, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.
* M. Li, X. Chen, S. Gao, S. Gao, Z. Chan, D. Zhao, and R. Yan (2020)VMSMO: learning to generate multimodal summary for video-based news articles. In EMNLP, Cited by: SS1, SS2.

Li, Y.; Song, Y.; Cao, L.; Tetreault, J.; Goldberg, L.; Jaimes, A.; and Luo, J. 2016. TGIF: A New Dataset and Benchmark on Animated GIF Description. In _CVPR_.
* Lin (2014) Lin, C.-Y. 2014. ROUGE: A Package for Automatic Evaluation of Summaries. In _ACL_.
* Liu et al. (2018) Liu, L.; Lu, Y.; Yang, M.; Qu, Q.; Zhu, J.; and Li, H. 2018. Generative Adversarial Network for Abstractive Text Summarization. In _AAAI_.
* Liu (2019) Liu, Y. 2019. Fine-tune BERT for Extractive Summarization. In _arXiv:1903.10318_.
* Liu and Lapata (2019) Liu, Y.; and Lapata, M. 2019. Text Summarization with Pretrained Encoders. In _EMNLP-IJCNLP_.
* Liu et al. (2019) Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.; Levy, O.; Lewis, M.; Zettlemoyer, L.; and Stoyanov, V. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. In _arxiv:1907.11692_.
* Lloret et al. (2013) Lloret, E.; Roma-Ferri, M. T.; and Palomar, M. 2013. COMPENDIUM: A Text Summarization System for Generating Abstracts of Research Papers. In _Data & Knowledge Engineering_.
* Marchesotti et al. (2011) Marchesotti, L.; Perronnin, F.; Larlus, D.; and Csurka, G. 2011. VMSMO: Learning to Generate Multimodal Summary for Video-based News Articles. In _ICCV_.
* Microsoft (2021) Microsoft. 2021. Azure Cognitive Services. [https://reurl.cc/Qjqe45](https://reurl.cc/Qjqe45). Accessed: 2020-09-04.
* Narayan et al. (2018) Narayan, S.; Cohen, S. B.; and Lapata, M. 2018. Ranking Sentences for Extractive Summarization with Reinforcement Learning. In _NAACL_.
* Parveen, Mesgar, and Strube (2016) Parveen, D.; Mesgar, M.; and Strube, M. 2016. Generating Coherent Summaries of Scientific Articles Using Coherence Patterns. In _EMNLP_.
* Paulus, Xiong, and Socher (2018) Paulus, R.; Xiong, C.; and Socher, R. 2018. A Deep Reinforced Model for Abstractive Summarization. In _ICLR_.
* Ren et al. (2015) Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In _NeurIPS_.
* Rush, Chopra, and Weston (2015) Rush, A. M.; Chopra, S.; and Weston, J. 2015. A Neural Attention Model for Abstractive Sentence Summarization. In _EMNLP_.
* See, Liu, and Manning (2017) See, A.; Liu, P. J.; and Manning, C. D. 2017. Get To The Point: Summarization with Pointer-Generator Networks. In _ACL_.
* Sefid and Wu (2019) Sefid, A.; and Wu, J. 2019. Automatic Slide Generation for scientific Papers. In _K-CAP_.
* Song and Soleymani (2019) Song, Y.; and Soleymani, M. 2019. Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval. In _CVPR_.
* Suzuki and Abe (1985) Suzuki, S.; and Abe, K. 1985. Topological Structural Analysis of Digitized Images by Border Following. In _CVGIP_.
* Vendrov et al. (2016) Vendrov, I.; Kiros, R.; Fidler, S.; and Urtasun, R. 2016. Order-Embeddings of Images and Language. In _ICLR_.
* Vinyals et al. (2016) Vinyals, O.; Toshev, A.; Bengio, S.; and Erhan, D. 2016. Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge. In _TPAMI_.
* Williams and Zipser (1989) Williams, R. J.; and Zipser, D. 1989. A Learning Algorithm for Continually Running Fully Recurrent Neural Networks. In _Neural computation_.
* Xu et al. (2016) Xu, J.; Mei, T.; Yao, T.; and Rui, Y. 2016. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. In _CVPR_.
* Yasunaga et al. (2019) Yasunaga, M.; Kasai, J.; Zhang, R.; Fabbri, A. R.; Li, I.; Friedman, D.; and Radev, D. R. 2019. ScisummNet: A Large Annotated Corpus and Content-Impact Models for Scientific Paper Summarization with Citation Networks. In _AAAI_.
* Yasunaga et al. (2017) Yasunaga, M.; Zhang, R.; Meelu, K.; Pareek, A.; Srinivasan, K.; and Radev, D. 2017. Graph-based Neural Multi-Document Summarization. In _CoNLL_.
* Yin and Pei (2014) Yin, W.; and Pei, Y. 2014. Optimizing Sentence Modeling and Selection for Document Summarization. In _IJCAI_.
* You et al. (2016) You, Q.; Jin, H.; Wang, Z.; Fang, C.; and Luo, J. 2016. Image Captioning with Semantic Attention. In _CVPR_.
* Zhang et al. (2020) Zhang, J.; Zhao, Y.; Saleh, M.; and Liu, P. J. 2020. PE-GASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. In _ICML_.
* Zhu et al. (2019) Zhu, J.; Li, H.; Liu, T.; Zhou, Y.; Zhang, J.; and Zong, C. 2019. MSMO: Multimodal Summarization with Multimodal Output. In _EMNLP_.
* Zhu et al. (2020) Zhu, J.; Zhou, Y.; Zhang, J.; Li, H.; Zong, C.; and Li, C. 2020. Multimodal Summarization with Guidance of Multimodal Reference. In _AAAI_.