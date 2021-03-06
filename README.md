# Data-Mining-Experiments

A collection of data mining experiments done during the data\-mining course\.

## 1\. Douban\-Spider👻

A tiny spider that fetches a list of [Douban Movie Top250](https://movie.douban.com/top250)\. **No** further analysis however\.

## 2\. Iris\-Classification🎉

Iris can be divided into three different varieties according to its calyx and petal size\. This experiment is to show how to analyze data and build a classification model as a novice (me)\.

## 3\. Order\-From\-Tmall🛍

This experiment makes good use of `pyecharts` to make data visualization, using 3 datasets about Tmall (Double Eleven, wholesale)\. While cleaning data and building visualization, valuable information can be revealed from the datasets\. 

Finally, an **RFM** model is implemented to predict the most valuable customers\.

> RFM means:
>
> - R-Recency, last purchase time 
> - F-Frequency, consumption frequency
> - M-Monetary, consumption amount

## 4\. Closed\-Companies📉

Learn how to draw word cloud\.

## 5\. Chinese\-News\-Digest\-Classification📣

This experiment implements **CNN (Convolutional Neural Networks)** to make inference using a dataset of 56821 pieces of Chinese news digest fetched from websites\. The purpose of the model is to classify sentences into one of ten categories\.

## 6\. Malicious-URL😈

With this analysis, we explored some important features that have proved to be sound and effective in predicting phishing/malicious websites based on lexical characteristics of URL.

We have 2 datasets: **top-1m.csv** (most visited domains) and **phishing_verified_online.csv** as a collection of malicious URLs (where domains can be duplicated)\.

## 7\. XSS\-Detection💀

Detect XSS strings using word2vec and MLP (Multi\-Layer Perceptron)\. 

**xssed.csv** is a classic data set of XSS codes\. 

## 8\. DGA\-Detection🎃

> Domain generation algorithms (DGA) are algorithms seen in various families of malware that are used to periodically generate a large number of domain names that can be used as rendezvous points with their command and control servers. The large number of potential rendezvous points makes it difficult for law enforcement to effectively shut down botnets, since infected computers will attempt to contact some of these domain names every day to receive updates or commands. The use of public-key cryptography in malware code makes it unfeasible for law enforcement and other actors to mimic commands from the malware controllers as some worms will automatically reject any updates not signed by the malware controllers.

Use N\-Gram and SVM (not so suitable for this task)\.

## 9\. Webshell\-Detection🙄

Detect PHP webshell using **Naive\-Bayes** and **Random Forest**\.

