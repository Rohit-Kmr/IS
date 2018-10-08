**KMEANS IDS**

** REQUIREMENT **
----------------------------------------------------------------------------------------------------------------------------------------

TO RUN
  1. install ancaconda navigator
  2. run the file in jupyter notebook

NOTE: please change the file name When using Currently set to "2015/01/20150101.txt"

NOTE2: Methodology/process is written at the end of file

** DATA STRUCTURE USED **
-----------------------------------------------------------------------------------------------------------------------------------------

LIST : it is a set values

class SelectKBest(_BaseFilter)
 |  Select features according to the k highest scores.
 |  
 |  Read more in the :ref:`User Guide <univariate_feature_selection>`.
 |  
 |  Parameters
 |  ----------
 |  score_func : callable
 |      Function taking two arrays X and y, and returning a pair of arrays
 |      (scores, pvalues) or a single array with scores.
 |      Default is f_classif (see below "See also"). The default function only
 |      works with classification tasks.
 |  
 |  k : int or "all", optional, default=10
 |      Number of top features to select.
 |      The "all" option bypasses selection, for use in a parameter search.
 |  
 |  Attributes
 |  ----------
 |  scores_ : array-like, shape=(n_features,)
 |      Scores of features.
 |  
 |  pvalues_ : array-like, shape=(n_features,)
 |      p-values of feature scores, None if `score_func` returned only scores.
 |
 |  __init__(self, score_func=<function f_classif at 0x000002D277D1E8C8>, k=10)
 |      Initialize self.
 |
 | fit_transform(self, X, y=None, **fit_params)
 |      Fit to data, then transform it.
 |      
 |      Fits transformer to X and y with optional parameters fit_params
 |      and returns a transformed version of X.
 |      
 |      Parameters
 |      ----------
 |      X : numpy array of shape [n_samples, n_features]
 |          Training set.
 |      
 |      y : numpy array of shape [n_samples]
 |          Target values.
 |      
 |      Returns
 |      -------
 |      X_new : numpy array of shape [n_samples, n_features_new]
 |          Transformed array.


class DataFrame(pandas.core.generic.NDFrame)
 |  Two-dimensional size-mutable, potentially heterogeneous tabular data
 |  structure with labeled axes (rows and columns). Arithmetic operations
 |  align on both row and column labels. Can be thought of as a dict-like
 |  container for Series objects. The primary pandas data structure.
 |  
 |  Parameters
 |  ----------
 |  data : numpy ndarray (structured or homogeneous), dict, or DataFrame
 |      Dict can contain Series, arrays, constants, or list-like objects
 |  
 |      .. versionchanged :: 0.23.0
 |         If data is a dict, argument order is maintained for Python 3.6
 |         and later.
 |  
 |  index : Index or array-like
 |      Index to use for resulting frame. Will default to RangeIndex if
 |      no indexing information part of input data and no index provided
 |  columns : Index or array-like
 |      Column labels to use for resulting frame. Will default to
 |      RangeIndex (0, 1, 2, ..., n) if no column labels are provided
 |  dtype : dtype, default None
 |      Data type to force. Only a single dtype is allowed. If None, infer
 |  copy : boolean, default False
 |      Copy data from inputs. Only affects DataFrame / 2d ndarray input
 |  
 |  Examples
 |  --------
 |  Constructing DataFrame from a dictionary.
 |  
 |  >>> d = {'col1': [1, 2], 'col2': [3, 4]}
 |  >>> df = pd.DataFrame(data=d)
 |  >>> df
 |     col1  col2
 |  0     1     3
 |  1     2     4
 |  
 |  Notice that the inferred dtype is int64.
 |  
 |  >>> df.dtypes
 |  col1    int64
 |  col2    int64
 |  dtype: object
 |  
 |  To enforce a single dtype:
 |  
 |  >>> df = pd.DataFrame(data=d, dtype=np.int8)
 |  >>> df.dtypes
 |  col1    int8
 |  col2    int8
 |  dtype: object


class KMeans(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin, sklearn.base.TransformerMixin)
 |  K-Means clustering
 |  
 |  Read more in the :ref:`User Guide <k_means>`.
 |  
 |  Parameters
 |  ----------
 |  
 |  n_clusters : int, optional, default: 8
 |      The number of clusters to form as well as the number of
 |      centroids to generate.
 |  
 |  init : {'k-means++', 'random' or an ndarray}
 |      Method for initialization, defaults to 'k-means++':
 |  
 |      'k-means++' : selects initial cluster centers for k-mean
 |      clustering in a smart way to speed up convergence. See section
 |      Notes in k_init for more details.
 |  
 |      'random': choose k observations (rows) at random from data for
 |      the initial centroids.
 |  
 |      If an ndarray is passed, it should be of shape (n_clusters, n_features)
 |      and gives the initial centers.
 |  
 |  n_init : int, default: 10
 |      Number of time the k-means algorithm will be run with different
 |      centroid seeds. The final results will be the best output of
 |      n_init consecutive runs in terms of inertia.
 |  
 |  max_iter : int, default: 300
 |      Maximum number of iterations of the k-means algorithm for a
 |      single run.
 |  
 |  tol : float, default: 1e-4
 |      Relative tolerance with regards to inertia to declare convergence
 |  
 |  precompute_distances : {'auto', True, False}
 |      Precompute distances (faster but takes more memory).
 |  
 |      'auto' : do not precompute distances if n_samples * n_clusters > 12
 |      million. This corresponds to about 100MB overhead per job using
 |      double precision.
 |  
 |      True : always precompute distances
 |  
 |      False : never precompute distances
 |  
 |  verbose : int, default 0
 |      Verbosity mode.
 |  
 |  random_state : int, RandomState instance or None, optional, default: None
 |      If int, random_state is the seed used by the random number generator;
 |      If RandomState instance, random_state is the random number generator;
 |      If None, the random number generator is the RandomState instance used
 |      by `np.random`.
 |  
 |  copy_x : boolean, default True
 |      When pre-computing distances it is more numerically accurate to center
 |      the data first.  If copy_x is True, then the original data is not
 |      modified.  If False, the original data is modified, and put back before
 |      the function returns, but small numerical differences may be introduced
 |      by subtracting and then adding the data mean.
 |  
 |  n_jobs : int
 |      The number of jobs to use for the computation. This works by computing
 |      each of the n_init runs in parallel.
 |  
 |      If -1 all CPUs are used. If 1 is given, no parallel computing code is
 |      used at all, which is useful for debugging. For n_jobs below -1,
 |      (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
 |      are used.
 |  
 |  algorithm : "auto", "full" or "elkan", default="auto"
 |      K-means algorithm to use. The classical EM-style algorithm is "full".
 |      The "elkan" variation is more efficient by using the triangle
 |      inequality, but currently doesn't support sparse data. "auto" chooses
 |      "elkan" for dense data and "full" for sparse data.
 |  
 |  Attributes
 |  ----------
 |  cluster_centers_ : array, [n_clusters, n_features]
 |      Coordinates of cluster centers
 |  
 |  labels_ :
 |      Labels of each point
 |  
 |  inertia_ : float
 |      Sum of squared distances of samples to their closest cluster center.
 |
 |  Attributes
 | ------------
 |  __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  fit(self, X, y=None)
 |      Compute k-means clustering.
 |      
 |      Parameters
 |      ----------
 |      X : array-like or sparse matrix, shape=(n_samples, n_features)
 |          Training instances to cluster.
 |      
 |      y : Ignored
 |  
 |  predict(self, X)
 |      Predict the closest cluster each sample in X belongs to.
 |      
 |      In the vector quantization literature, `cluster_centers_` is called
 |      the code book and each value returned by `predict` is the index of
 |      the closest code in the code book.
 |      
 |      Parameters
 |      ----------
 |      X : {array-like, sparse matrix}, shape = [n_samples, n_features]
 |          New data to predict.
 |      
 |      Returns
 |      -------
 |      labels : array, shape [n_samples,]
 |          Index of the cluster each sample belongs to. 


** FUNCTIONS USED **
----------------------------------------------------------------------------------------------------------------------------------------

 1) normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
    Scale input vectors individually to unit norm (vector length).
    
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.
    
    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    
    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.
    
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).
    
    return_norm : boolean, default False
        whether to return the computed norms
    
    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.
    
    norms : array, shape [n_samples] if axis=1 else [n_features]
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

 2) fit_transform(self, X, y=None, **fit_params)
 	    Fit to data, then transform it.
      
      Fits transformer to X and y with optional parameters fit_params
      and returns a transformed version of X.
      
      Parameters
      ----------
      X : numpy array of shape [n_samples, n_features]
          Training set.
      
      y : numpy array of shape [n_samples]
          Target values.
      
      Returns
      -------
      X_new : numpy array of shape [n_samples, n_features_new]
          Transformed array.


 3) train_test_split(*arrays, **options)
    Split arrays or matrices into random train and test subsets
    
    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    
    Read more in the :ref:`User Guide <cross_validation>`.
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    
    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    
    train_size : float, int, or None, default None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    
    stratify : array-like or None (default is None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

 4) fit(self, X, y=None)
       Compute k-means clustering.
       
       Parameters
       ----------
       X : array-like or sparse matrix, shape=(n_samples, n_features)
           Training instances to cluster.
       
       y : Ignored

 5) axes(arg=None, **kwargs)
    Add an axes to the current figure and make it the current axes.
    
    Parameters
    ----------
    arg : None or 4-tuple or Axes
        The exact behavior of this function depends on the type:
    
        - *None*: A new full window axes is added using
          ``subplot(111, **kwargs)``
        - 4-tuple of floats *rect* = ``[left, bottom, width, height]``.
          A new axes is added with dimensions *rect* in normalized
          (0, 1) units using `~.Figure.add_axes` on the current figure.
        - `.Axes`: This is equivalent to `.pyplot.sca`. It sets the current
          axes to *arg*. Note: This implicitly changes the current figure to
          the parent of *arg*.
    
          .. note:: The use of an Axes as an argument is deprecated and will be
                    removed in v3.0. Please use `.pyplot.sca` instead.
    
    Other Parameters
    ----------------
    **kwargs :
        For allowed keyword arguments see `.pyplot.subplot` and
        `.Figure.add_axes` respectively. Some common keyword arguments are
        listed below:
    
        ========= =========== =================================================
        kwarg     Accepts     Description
        ========= =========== =================================================
        facecolor color       the axes background color
        frameon   bool        whether to display the frame
        sharex    otherax     share x-axis with *otherax*
        sharey    otherax     share y-axis with *otherax*
        polar     bool        whether to use polar axes
        aspect    [str | num] ['equal', 'auto'] or a number.  If a number, the
                              ratio of y-unit/x-unit in screen-space.  See also
                              `~.Axes.set_aspect`.
        ========= =========== =================================================
    
    Returns
    -------
    axes : Axes
        The created or activated axes.
  
 6) scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs) method of matplotlib.axes._subplots.Axes3DSubplot instance
    Create a scatter plot.
    
    ============  ========================================================
    Argument      Description
    ============  ========================================================
    *xs*, *ys*    Positions of data points.
    *zs*          Either an array of the same length as *xs* and
                  *ys* or a single value to place all points in
                  the same plane. Default is 0.
    *zdir*        Which direction to use as z ('x', 'y' or 'z')
                  when plotting a 2D set.
    *s*           Size in points^2.  It is a scalar or an array of the
                  same length as *x* and *y*.
    
    *c*           A color. *c* can be a single color format string, or a
                  sequence of color specifications of length *N*, or a
                  sequence of *N* numbers to be mapped to colors using the
                  *cmap* and *norm* specified via kwargs (see below). Note
                  that *c* should not be a single numeric RGB or RGBA
                  sequence because that is indistinguishable from an array
                  of values to be colormapped.  *c* can be a 2-D array in
                  which the rows are RGB or RGBA, however, including the
                  case of a single row to specify the same color for
                  all points.
    
    *depthshade*
                  Whether or not to shade the scatter markers to give
                  the appearance of depth. Default is *True*.
    ============  ========================================================
    
    Keyword arguments are passed on to
    :func:`~matplotlib.axes.Axes.scatter`.
    
    Returns a :class:`~mpl_toolkits.mplot3d.art3d.Patch3DCollection`

 7) index(...) method of builtins.list instance
    L.index(value, [start, [stop]]) -> integer -- return first index of value.
    Raises ValueError if the value is not present.

 8) predict(self, X)
       Predict the closest cluster each sample in X belongs to.
       
       In the vector quantization literature, `cluster_centers_` is called
       the code book and each value returned by `predict` is the index of
       the closest code in the code book.
       
       Parameters
       ----------
       X : {array-like, sparse matrix}, shape = [n_samples, n_features]
           New data to predict.
       
       Returns
       -------
       labels : array, shape [n_samples,]
           Index of the cluster each sample belongs to. 

 9) confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    Compute confusion matrix to evaluate the accuracy of a classification
    
    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.
    
    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
    
    Read more in the :ref:`User Guide <confusion_matrix>`.
    
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    
    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix
10) ravel(...) method of numpy.ndarray instance
    a.ravel([order])
    
    Return a flattened array.


**METHODOLOGY**
----------------------------------------------------------------------------------------------------------------------------------------

A. IN this we first do some preprocessing steps
	1. convert catagorical values to metric
	2. convert IP address to decimal
	3. convert time to metric
	4. convert malware and other detection to 0 and 1(detected)
	5. convert label(i.e our target) to 0( no attack) and 1(any attack)

B. Due to huge difference in range we then normalize the data

C. Now we select best 3 metric from the data using chi square method( which is one of the best for feature selection in sparse data set)
	and plot it using scatter3d()

D. Divide the data into training set( 80% ) and test set ( 20% )

E. Now we apply K-Means on training set

F. now we find the label of each cluster

G. now we predict the labels of test set

H. Now we create a confusion matrix and use it to find Accuracy, Precision, Recall, False Positive Rate, False Negative rate
