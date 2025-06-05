## linkedin post sentiment analysis

analyzes linkedin posts using textblob sentiment analysis and lda topic modeling.

### what it does
- combines json files from linkedin posts
- cleans and processes text data
- analyzes sentiment and topics
- creates visualizations

### setup
```bash
pip install (all the important libraries like matplotlib, sklearn, texblob, nltk, seaborn, etc)
```

### usage
```bash
python post-analysis.py
```

### requirements
- python 3.x
- pandas
- numpy 
- nltk
- textblob
- scikit-learn
- matplotlib
- seaborn

### files
- input: `linkedin-post/*.json`
- output: `final-dataset.csv`
- plots: `plots/`