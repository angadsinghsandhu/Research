# Rebuttal to Review 3

We value the in-depth examination of our research and the perceptive remarks you shared regarding the existing knowledge in this specific field.

## Issue #1

First, we would like to address our methodology. Here is a flowchart to make the process proposed in the paper more clear as well as a excerpt on how K-means is applied in this context:

![methodology flow](https://github.com/angadsinghsandhu/Research/blob/main/Topic%20Modeling%20Trends/images/methodology_flow.jpg)

In our process, we begin with a corpus of text data. As a preliminary step, we partition this corpus based on distinct time periods to facilitate the temporal aspect of our analysis. With these partitioned datasets, Word2Vec is utilized in constructing separate vector spaces for each time period. This transformation into a vector space enables us to explore similarities and differences among words in different time periods. To unravel semantic relationships, we employ K-means clustering, an algorithm that aids in discerning patterns of similarity. The Gensim library allows all these tools to be used simultaneously, encapsulating the entire process — embedding the words, clustering them using K-means, and allowing us to query for similarity. It is important to note that K-means' role in this process is to allow for similarity analysis between words.

## Issue #2

As this is a short paper, we unfortunately had to take away a lot of our examples to fit the space. To make concurrent points on how this method produces concurrent results, we are attaching further case studies to drive our point across.

![Trends of relative term "cdc" with respect to "unofficial"](https://github.com/angadsinghsandhu/Research/blob/main/Topic%20Modeling%20Trends/images/1_example.png)

| Months | Jan'20 | Jun'20 | Dec'20 | Jan'21 | Jun'21 | Dec'21 |
|--------|--------|--------|--------|--------|--------|--------|
| CDC    | 0.00   | 0.262  | 0.123  | 0.116  | 0.109  | 0.112  |

The above trend is created using ***Twitter data*** from `2020` to `2021` (divided in time periods of 4 months) to study the trends of the public discourse around Covid and other issues. Here, the trem `"CDC"` (Center for Disease Control) is our **relative terms** that forms the trends with respect to the term `"Unofficial"` as the **base term**.

As it can be deduced from the data, the term `"CDC"` starts being considered from June 2020, corresponding with the sharp increase in COVID-19 cases. A comparatively *large cosine distance* can be observed between `"CDC"` and `"Unofficial"`, conveying that the organization was held in *high regard* as an entity to look for guidance and information in such a situation.

As time progresses to **June and December**, the public's trust in the "Media" and government organizations deteriorate with the number of *increasing deaths* and growth in the *anti-vaccination agenda* as well as an increased support from *pro-vaccination groups*. This leads to the **variance** in cosine distance, hence showcasing the growing distrust in the organization.

![Trend of the terms "CDC", "Fauci" and "experts" through 2020-21 with the base term "Trust"](https://github.com/angadsinghsandhu/Research/blob/main/Topic%20Modeling%20Trends/images/2_cs_covid.png)

| Months  | Jan'20 | Jun'20 | Dec'20 | Jan'21 | Jun'21 | Dec'21 |
|---------|--------|--------|--------|--------|--------|--------|
| CDC     | 0.00   | 0.413  | 0.132  | 0.128  | 0.103  | 0.114  |
| Fauci   | 0.00   | 0.521  | 0.187  | 0.111  | 0.139  | 0.153  |
| Experts | 0.280  | 0.591  | 0.133  | 0.126  | 0.146  | 0.109  |

The following case study is a perfect example to signify the discrepancies observed by quantizing such arbitrary concepts such as *trust* and *formalism* of an organization. Even though the terms `"Unofficial"` and `"Trust"` may seem different semantically, both of them produce similar trends (but not exact!! each base word creates new trends for all relative words) when in relation to `"CDC"`. This may be attributed to the absence of a general consensus.

An important fact that can be observed in both cases is the large difference in the values of the terms between *June 2020 to December 2020*. This corresponds to a greater shift in the political landscape as well as the public opinion. Where one school of thought went from considering CDC from *competent to inept*, in contrast, another may have started considering the said organization as more *trustworthy*. But, the COVID-19 case study also displays that all the relative terms (that are closely correlated to each-other) such as CDC, `Dr. Anthony Fauci` (Chief Medical Advisor to the President of the United States) and `Experts` all show **similar trends**.

## Issue #3

We are also specifying how our method performs better than `Google Trends` the go-to method for the average person to see how trends are forming. We use that tool as the baseline comparison for our study. Where `Google Trends` that forms trends by stochastically counting the reference sof terms over time, not considering any semantic data, hence forming wrong trends oftenly.

> Google has a popular trends tool to show how words/topics are being used by its users, which is the baseline of our study. It only stochastically counts how many times a certain keyword is mentioned, and therefore a single keyword that is being used in *different contexts* can be **easily misinterpreted** when analyzing **multiple topics**. For example, `“race”` in the context of the `“Olympics”` means something very different than `“race”` in the context of `“government”`, but `Google trends` shows the **same** trendline for both of these separate phenomena.

Hence,

> We propose a **novel** method where instead of individually looking at each term we rather look at it semantically by comparing it with **other words** being used in its context, i.e the **base** word. Therefore, our method has the *capability to uncover trends across varying contexts*. Trends established through the application of embeddings and cosine disatnce along with k-means clustering to the term `"race"`, with `"Olympics"` as the **base term**, ***diverge significantly*** from the trends observed when the term `"race"` is assessed relative to the base term `"government"`. Thus, our methodology differentiates itself from other forms of trend detection.

Since this is an **unsupervised task** there are no test and validation labels that may be crossed referenced to the problem at hand. But in our future work we will provide more metrics and case studies to make sure the evidence of our research is unquestionable.

Also look into our rebuttal to `5uNc` (Reviewer #1) and the The paper mentioned by them.
