# ADDRESSING ANOMALY DETECTION IN HIGH VOLUME SETTING

MY MS (Data Science) Thesis Code Base - (k173004 FAST NUCES Karachi).
---------------------------------------------------------------------

## Datasets:

AML dataset: Anti-money laundering dataset comprises of SAR (suspicious activity report) and Non-SAR transactions of the individuals in a financial institution accumulated for the period of 1 year.
    
Home Credit Default dataset: Home credit default contain the transactional and repayment history of the customers which are provided loan with no credit history. (Ref: https://www.kaggle.com/c/home-credit-default-risk/data)
    
Credit card fraud dataset: This dataset contain the fraudulent and non-fraudulent credit card transactions of all the customers for a period of one month which are European credit card holders. (Ref: https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3)
    
Paysim: The dataset comprises of simulated mobile money transactions based real transactions extracted from one month of financial logs which consists of around 6 million transactions out of which 8312 transactions are labeled as fraud. (Ref: https://www.kaggle.com/ntnu-testimon/paysim1/version/1)
    
CIFAR: This dataset consists of 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. (Ref: https://www.cs.toronto.edu/~kriz/cifar.html)

## Experiments: 

The experiments are categorized into 3 types: 
Cramer Rao Lower Bound, No Penalty and L2 penalty based on multiple batch settings e.g., 5 and 20 batches for each data-set. 

Cramer Rao Lower Bound Optimization is one of artifacts of the proposed model in this study which ensures the information retention of the previous batches when training on the current batch, however the competing techniques of optimization from the literature were also compared namely SGD in which gradient steps according to the current batch only and will minimize the loss of the current batch but destroy information of the previous batch and L2 penalty which is too severe that it tries to retain the information of  previous batch only at the expense of not learning current batch data distribution. The experiments are carried out using multiple batch sizes as to validate the technique on different variances among the batches. Also in case of online setting, the technique is validated in comparison of the prevailing techniques.

The evaluation metrics used in this experimental setup is accuracy which is measured as correctly predicted to the total predictions in the dataset.

## Results:

CRLBO outperforms other techniques on a tabular dataset in 20 batch setting by 18\%. This can be attributed to the high inter-batch variance present in the dataset. Thus it can be inferred that CRLBO retains maximum information on tabular datasets in high variant settings.

CRLBO outperforms other techniques by 19\% on high dimensional images in online settings. Thus it can be inferred that CRLBO retains maximum information in high dimensional settings.

The inference obtained from this investigation is that in high volume settings, if the distribution of the data is similar then any state of the art technique could achieve comparable generalization however in case of varying distribution and high variance across batches, CRLBO  achieves best generalization by adjusting the weights in a way that it also retains the information of the previous batches which makes this technique work better over state of the art prevailing techniques. 


