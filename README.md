# Thambawita 2018
The code to show execution and the results of the paper titled "The Medico-Task 2018: Disease Detection in the Gastrointestinal Tract using Global Features and Deep Learning" a working note paper of Media Eval 2018.
The algorithm is executed on the dataset of Media Eval with the confusion matrices as shown below:

## MediaEval 2018
The confusion matrix for the Media Eval 2018 task using the discussed algorithm Run 5 is as follows with the accuracy of 0.922 and the f1 easure of 0.908:

| Confusion Matrix | retroflex-rectum  | out-of-patient | ulcerative-colitis | normal-cecum | normal-z-line | dyed-lifted-polyps | blurry-nothing | retroflex-stomach | instruments | dyed-resection-margins | stool-plenty | esophagitis | normal-pylorus | polyps | stool-inclusions | colon-clear |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| retroflex-rectum | 183 | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 4 | 0 | 0 |
| out-of-patient | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ulcerative-colitis | 1 | 0 | 502 | 11 | 0 | 1 | 1 | 0 | 0 | 0 | 2 | 2 | 6 | 16 | 0 | 0 |
| normal-cecum | 0 | 0 | 7 | 571 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| normal-z-line | 0 | 0 | 0 | 0 | 469 | 0 | 0 | 1 | 0 | 0 | 0 | 88 | 4 | 1 | 0 | 0 |
| dyed-lifted-polyps | 1 | 0 | 1 | 1 | 0 | 499 | 0 | 0 | 0 | 49 | 0 | 0 | 0 | 5 | 0 | 0 |
| blurry-nothing | 0 | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| retroflex-stomach | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 396 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| instruments | 2 | 0 | 32 | 9 | 0 | 65 | 0 | 0 | 38 | 22 | 3 | 0 | 0 | 101 | 0 | 1 |
| dyed-resection-margins | 0 | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 524 | 0 | 0 | 0 | 0 | 0 | 0 |
| stool-plenty | 0 | 0 | 14 | 12 | 0 | 0 | 1 | 0 | 0 | 0 | 1932 | 0 | 1 | 3 | 2 | 0 |
| esophagitis | 0 | 0 | 0 | 0 | 136 | 0 | 0 | 0 | 0 | 0 | 0 | 416 | 4 | 0 | 0 | 0 |
| normal-pylorus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 560 | 0 | 0 | 0 |
| polyps | 1 | 0 | 9 | 16 | 1 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 342 | 0 | 0 |
| stool-inclusions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 498 | 8 |
| colon-clear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1064 |

## MediaEval 2017
The confusion matrix for the Media Eval 2017 task using the discussed algorithm Run 5 is as follows with the accuracy of Y and the f1 easure of X:

| Confusion Matrix | ulcerative-colitis | normal-cecum | normal-z-line | dyed-lifted-polyps | dyed-resection-margins | esophagitis | normal-pylorus | polyps |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ulcerative-colitis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| normal-cecum | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| normal-z-line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| dyed-lifted-polyps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| dyed-resection-margins | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| esophagitis | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| normal-pylorus | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| polyps | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |

