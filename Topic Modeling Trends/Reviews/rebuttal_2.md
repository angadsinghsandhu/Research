# Reubttal to Review 2

We appreciate your thorough analysis of our research work and the insightful comments you made on the body of knowledge in this domain.

## Issue #1

Here, the process **only** requires the end user to:

1. select the data on which the trend should be formed. (specifying the number of time periods)
2. a single base term.
3. at least one or more relative terms.

All further steps are handled by the process itself, given the user knows the number of timesteps (per month in our study) on which the trend should be formed.

## Issue #2

Our paper describes a preliminary set of promising results that can be improved by incorporating more current techniques that utilize attention to give us a finer resolution on embeddings. We specifically mention in our “Future Works” section our intention to advance the system in this regard.

Our “Future Works” section states that the next step to our research is using advanced techniques such as *Transformer-based models* to capture word-level attention to create the embeddings, but **does not** take away from the importance of this study, which will build the ***foundation*** for more *advanced* techniques.

## Issue #3

As this is a short paper, we unfortunately had to take away a lot of our examples to fit the space. To make concurrent points on how this method produces concurrent results, we are attaching further case studies to drive our point across.

![Trends of relative term "cdc" with respect to "official"](https://github.com/angadsinghsandhu/Research/blob/main/Topic%20Modeling%20Trends/images/1_example.png)

[Table 1]

As it can be deduced from the data, the term "CDC" starts being considered from June 2020, corresponding with the sharp increase in COVID-19 cases. A comparatively large cosine distance can be observed between "CDC" and "Unofficial", conveying that the organization was held in high regard as an entity to look for guidance and information in such a situation. As time progresses to June and December, the public's trust in the "Media" and government organizations deteriorate with the number of increasing deaths and growth in the anti-vaccination agenda as well as an increased support from pro-vaccination groups. This leads to the variance in cosine distance, hence showcasing the growing distrust in the organization.

[Table 2]
[use 2_cs_covid.png]
The following case study is a perfect example to signify the discrepancies observed by quantizing such arbitrary concepts such as trust and formalism of an organization. Even though the terms "Unofficial" and "Trust" may seem different semantically, both of them produce similar trends when in relation to "CDC". This may be attributed to the absence of a general consensus. 

An important fact that can be observed in both cases is the large difference in the values of the terms between June 2020 to December 2020. This corresponds to a greater shift in the political landscape as well as the public opinion. Where one school of thought went from considering CDC from competent to inept, in contrast, another may have started considering the said organization as more trustworthy. But, the COVID-19 case study also displays that all the relative terms (that are closely correlated to each-other) such as CDC, Dr. Anthony Fauci (Chief Medical Advisor to the President of the United States) and Experts all show similar trends.


We are also specifying how our method performs better than google trends the go-to method for the average person to see how trends are forming.

Google has trends to show how words/topics are being used by the world, which is the baseline of our study. It only stochastically counts how many times a certain keyword is mentioned, and therefore a single keyword that is being used in different contexts can be easily misinterpreted when analyzing multiple topics. For Example, “race” in the context of the “Olympics” means something very different than “race” in the context of “government,” but Google trends shows the same trendline for both of these separate phenomena. Hence, we propose a novel method where instead of individually looking at each term we rather look at it semantically by comparing it with other words being used in its context, i.e the base word. Therefore, our method has the capability to uncover trends across varying contexts. Trends established through the application of k-means clustering to the term "race," with "Olympics" as the base term, diverge significantly from the trends observed when the term "race" is assessed relative to the base term "government." Thus, our methodology differentiates itself from other forms of trend detection.

The output of our system results in varying degrees of convergence (or divergence) for a set of words. We believe utilizing a line plot best demonstrates this phenomena between trends where a user is able to see temporal change while simultaneously making comparisons between other words. In our early tests, using more complicated plots such as scatter plots or contour plots proved to be more difficult to comprehend, especially when the number of relative terms increased.

We have already shown various case studies to show concurrent results in our study and we hope to also show inter-community and intra-community trends formed within the same dataset.
