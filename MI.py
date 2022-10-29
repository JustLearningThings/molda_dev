'''
Created with love by Sigmoid
@Author - Butucea Adelina - butucea.adelina@gmail.com
'''

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted


from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.feature_extraction.text import CountVectorizer
FLOAT_DTYPES = (np.float16, np.float32, np.float64)

class MITransformer(BaseEstimator, TransformerMixin, _OneToOneFeatureMixin):
  def __init__(self, epsilon = 3e-10):
    self.epsilon = epsilon
  
  def fit(self, X, y=None):
    if not sp.issparse(X):
      X = sp.csr_matrix(X)
    return self
  
  def transform(self, X, y=None):
    if not sp.issparse(X):
      sp.csr_matrix(X)
    '''
    input: sparse matrix from the CountVectorizer instance
    output: processed matrix
    
    '''
    allsum = X.sum()
    rowsum = np.sum(X, axis=1)
    colsum = X.sum(axis=0)
    cat_freq = rowsum/allsum

    for i in range(len(rowsum)):
      for j in range(len(colsum)):
          X[i][j] = np.log((X[i][j]/(rowsum[i]+self.epsilon))/(X[i][j]/allsum*cat_freq[i]+self.epsilon)+self.epsilon)
    return X
  
  def fit_transform(self, X, y=None):
    return self.fit(X).transform(X)


class MIVectorizer(CountVectorizer):
    def __init__(
          self,
          *,
          input="content",
          encoding="utf-8",
          decode_error="strict",
          strip_accents=None,
          lowercase=True,
          preprocessor=None,
          tokenizer=None,
          analyzer="word",
          stop_words=None,
          token_pattern=r"(?u)\b\w\w+\b",
          ngram_range=(1, 1),
          max_cf=1.0,
          min_cf=1,
          max_features=None,
          vocabulary=None,
          binary=False,
          dtype=np.float64,
          epsilon=3e-10,
      ):

          super().__init__(
              input=input,
              encoding=encoding,
              decode_error=decode_error,
              strip_accents=strip_accents,
              lowercase=lowercase,
              preprocessor=preprocessor,
              tokenizer=tokenizer,
              analyzer=analyzer,
              stop_words=stop_words,
              token_pattern=token_pattern,
              ngram_range=ngram_range,
              max_df=max_cf,
              min_df=min_cf,
              max_features=max_features,
              vocabulary=vocabulary,
              binary=binary,
              dtype=dtype,
          )
          self.epsilon = epsilon

    def fit(self, raw_documents, y=None):
      self._mit = MITransformer(epsilon=self.epsilon)

      X = super().fit_transform(raw_documents)
      self._mit.fit(X)
      return self

    def transform(self, raw_documents):
      check_is_fitted(self, msg="The TF-ICF vectorizer is not fitted")
      X = super().fit_transform(raw_documents)
      return self._mit.transform(X.toarray(), copy=False)

    def fit_transform(self, raw_documents, y=None):
      self._mit = MITransformer(epsilon=self.epsilon)
      X = super().fit_transform(raw_documents)
      self._mit.fit(X)
      return self._mit.transform(X.toarray())
