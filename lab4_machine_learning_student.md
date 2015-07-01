
version 1.0.1
#![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# **Introduction to Machine Learning with Apache Spark**
## **Predicting Movie Ratings**
#### One of the most common uses of big data is to predict what users want.  This allows Google to show you relevant ads, Amazon to recommend relevant products, and Netflix to recommend movies that you might like.  This lab will demonstrate how we can use Apache Spark to recommend movies to a user.  We will start with some basic techniques, and then use the [Spark MLlib][mllib] library's Alternating Least Squares method to make more sophisticated predictions.
#### For this lab, we will use a subset dataset of 500,000 ratings we have included for you into your VM (and on Databricks) from the [movielens 10M stable benchmark rating dataset](http://grouplens.org/datasets/movielens/). However, the same code you write will work for the full dataset, or their latest dataset of 21 million ratings.
#### In this lab:
#### *Part 0*: Preliminaries
#### *Part 1*: Basic Recommendations
#### *Part 2*: Collaborative Filtering
#### *Part 3*: Predictions for Yourself
#### As mentioned during the first Learning Spark lab, think carefully before calling `collect()` on any datasets.  When you are using a small dataset, calling `collect()` and then using Python to get a sense for the data locally (in the driver program) will work fine, but this will not work when you are using a large dataset that doesn't fit in memory on one machine.  Solutions that call `collect()` and do local analysis that could have been done with Spark will likely fail in the autograder and not receive full credit.
[mllib]: https://spark.apache.org/mllib/

### Code
#### This assignment can be completed using basic Python and pySpark Transformations and Actions.  Libraries other than math are not necessary. With the exception of the ML functions that we introduce in this assignment, you should be able to complete all parts of this homework using only the Spark functions you have used in prior lab exercises (although you are welcome to use more features of Spark if you like!).


    import sys
    import os
    from test_helper import Test
    
    baseDir = os.path.join('data')
    inputPath = os.path.join('cs100', 'lab4', 'small')
    
    ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
    moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')

### **Part 0: Preliminaries**
#### We read in each of the files and create an RDD consisting of parsed lines.
#### Each line in the ratings dataset (`ratings.dat.gz`) is formatted as:
####   `UserID::MovieID::Rating::Timestamp`
#### Each line in the movies (`movies.dat`) dataset is formatted as:
####   `MovieID::Title::Genres`
#### The `Genres` field has the format
####   `Genres1|Genres2|Genres3|...`
#### The format of these files is uniform and simple, so we can use Python [`split()`](https://docs.python.org/2/library/stdtypes.html#str.split) to parse their lines.
#### Parsing the two files yields two RDDS
* #### For each line in the ratings dataset, we create a tuple of (UserID, MovieID, Rating). We drop the timestamp because we do not need it for this exercise.
* #### For each line in the movies dataset, we create a tuple of (MovieID, Title). We drop the Genres because we do not need them for this exercise.


    numPartitions = 2
    rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
    rawMovies = sc.textFile(moviesFilename)
    
    def get_ratings_tuple(entry):
        """ Parse a line in the ratings dataset
        Args:
            entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
        Returns:
            tuple: (UserID, MovieID, Rating)
        """
        items = entry.split('::')
        return int(items[0]), int(items[1]), float(items[2])
    
    
    def get_movie_tuple(entry):
        """ Parse a line in the movies dataset
        Args:
            entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
        Returns:
            tuple: (MovieID, Title)
        """
        items = entry.split('::')
        return int(items[0]), items[1]
    
    
    ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
    moviesRDD = rawMovies.map(get_movie_tuple).cache()
    
    ratingsCount = ratingsRDD.count()
    moviesCount = moviesRDD.count()
    
    print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)
    print 'Ratings: %s' % ratingsRDD.take(3)
    print 'Movies: %s' % moviesRDD.take(3)
    
    assert ratingsCount == 487650
    assert moviesCount == 3883
    assert moviesRDD.filter(lambda (id, title): title == 'Toy Story (1995)').count() == 1
    assert (ratingsRDD.takeOrdered(1, key=lambda (user, movie, rating): movie)
            == [(1, 1, 5.0)])

    There are 487650 ratings and 3883 movies in the datasets
    Ratings: [(1, 1193, 5.0), (1, 914, 3.0), (1, 2355, 5.0)]
    Movies: [(1, u'Toy Story (1995)'), (2, u'Jumanji (1995)'), (3, u'Grumpier Old Men (1995)')]


#### In this lab we will be examining subsets of the tuples we create (e.g., the top rated movies by users). Whenever we examine only a subset of a large dataset, there is the potential that the result will depend on the order we perform operations, such as joins, or how the data is partitioned across the workers. What we want to guarantee is that we always see the same results for a subset, independent of how we manipulate or store the data.
#### We can do that by sorting before we examine a subset. You might think that the most obvious choice when dealing with an RDD of tuples would be to use the [`sortByKey()` method][sortbykey]. However this choice is problematic, as we can still end up with different results if the key is not unique.
#### Note: It is important to use the [`unicode` type](https://docs.python.org/2/howto/unicode.html#the-unicode-type) instead of the `string` type as the titles are in unicode characters.
#### Consider the following example, and note that while the sets are equal, the printed lists are usually in different order by value, *although they may randomly match up from time to time.*
#### You can try running this multiple times.  If the last assertion fails, don't worry about it: that was just the luck of the draw.  And note that in some environments the results may be more deterministic.
[sortbykey]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sortByKey


    tmp1 = [(1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'delta')]
    tmp2 = [(1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'alpha')]
    
    oneRDD = sc.parallelize(tmp1)
    twoRDD = sc.parallelize(tmp2)
    oneSorted = oneRDD.sortByKey(True).collect()
    twoSorted = twoRDD.sortByKey(True).collect()
    print oneSorted
    print twoSorted
    assert set(oneSorted) == set(twoSorted)     # Note that both lists have the same elements
    assert twoSorted[0][0] < twoSorted.pop()[0] # Check that it is sorted by the keys
    assert oneSorted[0:2] != twoSorted[0:2]     # Note that the subset consisting of the first two elements does not match

    [(1, u'alpha'), (1, u'epsilon'), (1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]
    [(1, u'delta'), (1, u'epsilon'), (1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]


#### Even though the two lists contain identical tuples, the difference in ordering *sometimes* yields a different ordering for the sorted RDD (try running the cell repeatedly and see if the results change or the assertion fails). If we only examined the first two elements of the RDD (e.g., using `take(2)`), then we would observe different answers - **that is a really bad outcome as we want identical input data to always yield identical output**. A better technique is to sort the RDD by *both the key and value*, which we can do by combining the key and value into a single string and then sorting on that string. Since the key is an integer and the value is a unicode string, we can use a function to combine them into a single unicode string (e.g., `unicode('%.3f' % key) + ' ' + value`) before sorting the RDD using [sortBy()][sortby].
[sortby]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sortBy


    def sortFunction(tuple):
        """ Construct the sort string (does not perform actual sorting)
        Args:
            tuple: (rating, MovieName)
        Returns:
            sortString: the value to sort with, 'rating MovieName'
        """
        key = unicode('%.3f' % tuple[0])
        value = tuple[1]
        return (key + ' ' + value)
    
    
    print oneRDD.sortBy(sortFunction, True).collect()
    print twoRDD.sortBy(sortFunction, True).collect()

    [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]
    [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]


#### If we just want to look at the first few elements of the RDD in sorted order, we can use the [takeOrdered][takeordered] method with the `sortFunction` we defined.
[takeordered]: https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.takeOrdered


    oneSorted1 = oneRDD.takeOrdered(oneRDD.count(),key=sortFunction)
    twoSorted1 = twoRDD.takeOrdered(twoRDD.count(),key=sortFunction)
    print 'one is %s' % oneSorted1
    print 'two is %s' % twoSorted1
    assert oneSorted1 == twoSorted1

    one is [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]
    two is [(1, u'alpha'), (1, u'delta'), (1, u'epsilon'), (2, u'alpha'), (2, u'beta'), (3, u'alpha')]


### **Part 1: Basic Recommendations**
#### One way to recommend movies is to always recommend the movies with the highest average rating. In this part, we will use Spark to find the name, number of ratings, and the average rating of the 20 movies with the highest average rating and more than 500 reviews. We want to filter our movies with high ratings but fewer than or equal to 500 reviews because movies with few reviews may not have broad appeal to everyone.

#### **(1a) Number of Ratings and Average Ratings for a Movie**
#### Using only Python, implement a helper function `getCountsAndAverages()` that takes a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...)) and returns a tuple of (MovieID, (number of ratings, averageRating)). For example, given the tuple `(100, (10.0, 20.0, 30.0))`, your function should return `(100, (3, 20.0))`


    # TODO: Replace <FILL IN> with appropriate code
    
    # First, implement a helper function `getCountsAndAverages` using only Python
    def getCountsAndAverages(IDandRatingsTuple):
        """ Calculate average rating
        Args:
            IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
        Returns:
            tuple: a tuple of (MovieID, (number of ratings, averageRating))
        """
        ratingsList = list(IDandRatingsTuple[1])
        ratingsCount = len(ratingsList)
        ratingsAvg = sum(ratingsList)/float(ratingsCount)
        return (IDandRatingsTuple[0], (ratingsCount, ratingsAvg))


    # TEST Number of Ratings and Average Ratings for a Movie (1a)
    
    Test.assertEquals(getCountsAndAverages((1, (1, 2, 3, 4))), (1, (4, 2.5)),
                                'incorrect getCountsAndAverages() with integer list')
    Test.assertEquals(getCountsAndAverages((100, (10.0, 20.0, 30.0))), (100, (3, 20.0)),
                                'incorrect getCountsAndAverages() with float list')
    Test.assertEquals(getCountsAndAverages((110, xrange(20))), (110, (20, 9.5)),
                                'incorrect getCountsAndAverages() with xrange')

    1 test passed.
    1 test passed.
    1 test passed.


#### **(1b) Movies with Highest Average Ratings**
#### Now that we have a way to calculate the average ratings, we will use the `getCountsAndAverages()` helper function with Spark to determine movies with highest average ratings.
#### The steps you should perform are:
* #### Recall that the `ratingsRDD` contains tuples of the form (UserID, MovieID, Rating). From `ratingsRDD` create an RDD with tuples of the form (MovieID, Python iterable of Ratings for that MovieID). This transformation will yield an RDD of the form: `[(1, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e7c90>), (2, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e79d0>), (3, <pyspark.resultiterable.ResultIterable object at 0x7f16d50e7610>)]`. Note that you will only need to perform two Spark transformations to do this step.
* #### Using `movieIDsWithRatingsRDD` and your `getCountsAndAverages()` helper function, compute the number of ratings and average rating for each movie to yield tuples of the form (MovieID, (number of ratings, average rating)). This transformation will yield an RDD of the form: `[(1, (993, 4.145015105740181)), (2, (332, 3.174698795180723)), (3, (299, 3.0468227424749164))]`. You can do this step with one Spark transformation
* #### We want to see movie names, instead of movie IDs. To `moviesRDD`, apply RDD transformations that use `movieIDsWithAvgRatingsRDD` to get the movie names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form (average rating, movie name, number of ratings). This set of transformations will yield an RDD of the form: `[(1.0, u'Autopsy (Macchie Solari) (1975)', 1), (1.0, u'Better Living (1998)', 1), (1.0, u'Big Squeeze, The (1996)', 3)]`. You will need to do two Spark transformations to complete this step: first use the `moviesRDD` with `movieIDsWithAvgRatingsRDD` to create a new RDD with Movie names matched to Movie IDs, then convert that RDD into the form of (average rating, movie name, number of ratings). These transformations will yield an RDD that looks like: `[(3.6818181818181817, u'Happiest Millionaire, The (1967)', 22), (3.0468227424749164, u'Grumpier Old Men (1995)', 299), (2.882978723404255, u'Hocus Pocus (1993)', 94)]`


    # TODO: Replace <FILL IN> with appropriate code
    
    # From ratingsRDD with tuples of (UserID, MovieID, Rating) create an RDD with tuples of
    # the (MovieID, iterable of Ratings for that MovieID)
    movieIDsWithRatingsRDD = (ratingsRDD.map(lambda (x, y, z): (y, z)).groupByKey().mapValues(tuple))
    print 'movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3)
    
    # Using `movieIDsWithRatingsRDD`, compute the number of ratings and average rating for each movie to
    # yield tuples of the form (MovieID, (number of ratings, average rating))
    movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)
    print 'movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3)
    
    # To `movieIDsWithAvgRatingsRDD`, apply RDD transformations that use `moviesRDD` to get the movie
    # names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
    # (average rating, movie name, number of ratings)
    movieNameWithAvgRatingsRDD = (moviesRDD.join(movieIDsWithAvgRatingsRDD).map(lambda (m, (title, (cnt, avg))): (avg, title, cnt)))
    print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)

    movieIDsWithRatingsRDD: [(2, (1.0, 5.0, 4.0, 5.0, 4.0, 4.0, 2.0, 4.0, 4.0, 5.0, 3.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 4.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 3.0, 1.0, 4.0, 4.0, 3.0, 4.0, 2.0, 2.0, 2.0, 4.0, 3.0, 4.0, 3.0, 4.0, 4.0, 1.0, 1.0, 4.0, 3.0, 3.0, 5.0, 2.0, 3.0, 3.0, 5.0, 4.0, 3.0, 5.0, 1.0, 3.0, 3.0, 4.0, 2.0, 4.0, 4.0, 4.0, 3.0, 5.0, 3.0, 3.0, 2.0, 3.0, 4.0, 1.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 1.0, 4.0, 5.0, 3.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 2.0, 4.0, 2.0, 3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0, 2.0, 5.0, 4.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 1.0, 4.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 2.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0, 3.0, 2.0, 2.0, 3.0, 5.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 4.0, 2.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 1.0, 4.0, 1.0, 3.0, 4.0, 3.0, 3.0, 4.0, 1.0, 2.0, 4.0, 2.0, 1.0, 4.0, 3.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0, 1.0, 3.0, 2.0, 4.0, 2.0, 3.0, 4.0, 1.0, 4.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 1.0, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 2.0, 5.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0, 5.0, 4.0, 2.0, 4.0, 3.0, 3.0, 1.0, 3.0, 4.0, 3.0, 3.0, 3.0, 2.0, 5.0, 5.0, 4.0, 5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 4.0, 2.0, 3.0, 4.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.0, 2.0, 2.0, 4.0, 4.0, 1.0, 3.0, 5.0, 3.0, 4.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0)), (4, (4.0, 1.0, 3.0, 2.0, 1.0, 5.0, 4.0, 2.0, 3.0, 1.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 4.0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 1.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 3.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 5.0, 2.0, 1.0, 3.0, 2.0, 2.0, 1.0, 2.0, 2.0)), (6, (4.0, 4.0, 5.0, 5.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1.0, 4.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 2.0, 4.0, 4.0, 2.0, 5.0, 4.0, 4.0, 4.0, 2.0, 5.0, 3.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 5.0, 2.0, 4.0, 5.0, 5.0, 4.0, 4.0, 4.0, 5.0, 2.0, 5.0, 4.0, 4.0, 4.0, 3.0, 5.0, 5.0, 4.0, 3.0, 5.0, 3.0, 4.0, 4.0, 4.0, 5.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 3.0, 4.0, 2.0, 4.0, 4.0, 3.0, 3.0, 4.0, 3.0, 3.0, 5.0, 4.0, 4.0, 3.0, 5.0, 4.0, 4.0, 2.0, 4.0, 5.0, 4.0, 5.0, 4.0, 3.0, 5.0, 3.0, 4.0, 5.0, 4.0, 5.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 3.0, 4.0, 3.0, 3.0, 5.0, 1.0, 5.0, 5.0, 3.0, 4.0, 3.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 4.0, 2.0, 3.0, 3.0, 5.0, 4.0, 4.0, 3.0, 3.0, 5.0, 4.0, 5.0, 4.0, 3.0, 3.0, 4.0, 2.0, 2.0, 5.0, 3.0, 1.0, 3.0, 5.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0, 5.0, 5.0, 3.0, 3.0, 4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0, 4.0, 5.0, 4.0, 5.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 4.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 3.0, 3.0, 4.0, 2.0, 4.0, 5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 2.0, 5.0, 4.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 3.0, 5.0, 4.0, 5.0, 3.0, 5.0, 4.0, 2.0, 3.0, 4.0, 4.0, 5.0, 1.0, 5.0, 2.0, 3.0, 4.0, 2.0, 1.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 5.0, 3.0, 4.0, 4.0, 4.0, 5.0, 2.0, 3.0, 4.0, 4.0, 3.0, 5.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 3.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 5.0, 2.0, 3.0, 5.0, 4.0, 5.0, 4.0, 4.0, 4.0, 5.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 4.0, 1.0, 3.0, 4.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 5.0, 5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 4.0, 5.0, 3.0, 5.0, 4.0, 4.0, 3.0, 5.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, 3.0, 4.0, 3.0, 3.0, 5.0, 2.0, 4.0, 5.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 4.0, 5.0, 3.0, 5.0, 3.0, 4.0, 5.0, 2.0, 4.0, 5.0, 2.0, 5.0, 2.0, 5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 3.0, 1.0, 5.0, 2.0, 4.0, 5.0, 3.0, 5.0, 5.0, 4.0, 5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0, 2.0, 5.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0, 4.0, 4.0, 3.0, 5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 5.0, 4.0, 4.0, 5.0, 5.0, 4.0, 3.0))]
    
    movieIDsWithAvgRatingsRDD: [(2, (332, 3.174698795180723)), (4, (71, 2.676056338028169)), (6, (442, 3.7918552036199094))]
    
    movieNameWithAvgRatingsRDD: [(3.6818181818181817, u'Happiest Millionaire, The (1967)', 22), (3.0468227424749164, u'Grumpier Old Men (1995)', 299), (2.882978723404255, u'Hocus Pocus (1993)', 94)]
    



    # TEST Movies with Highest Average Ratings (1b)
    
    Test.assertEquals(movieIDsWithRatingsRDD.count(), 3615,
                    'incorrect movieIDsWithRatingsRDD.count() (expected 3615)')
    movieIDsWithRatingsTakeOrdered = movieIDsWithRatingsRDD.takeOrdered(3)
    Test.assertTrue(movieIDsWithRatingsTakeOrdered[0][0] == 1 and
                    len(list(movieIDsWithRatingsTakeOrdered[0][1])) == 993,
                    'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[0] (expected 993)')
    Test.assertTrue(movieIDsWithRatingsTakeOrdered[1][0] == 2 and
                    len(list(movieIDsWithRatingsTakeOrdered[1][1])) == 332,
                    'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[1] (expected 332)')
    Test.assertTrue(movieIDsWithRatingsTakeOrdered[2][0] == 3 and
                    len(list(movieIDsWithRatingsTakeOrdered[2][1])) == 299,
                    'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[2] (expected 299)')
    
    Test.assertEquals(movieIDsWithAvgRatingsRDD.count(), 3615,
                    'incorrect movieIDsWithAvgRatingsRDD.count() (expected 3615)')
    Test.assertEquals(movieIDsWithAvgRatingsRDD.takeOrdered(3),
                    [(1, (993, 4.145015105740181)), (2, (332, 3.174698795180723)),
                     (3, (299, 3.0468227424749164))],
                    'incorrect movieIDsWithAvgRatingsRDD.takeOrdered(3)')
    
    Test.assertEquals(movieNameWithAvgRatingsRDD.count(), 3615,
                    'incorrect movieNameWithAvgRatingsRDD.count() (expected 3615)')
    Test.assertEquals(movieNameWithAvgRatingsRDD.takeOrdered(3),
                    [(1.0, u'Autopsy (Macchie Solari) (1975)', 1), (1.0, u'Better Living (1998)', 1),
                     (1.0, u'Big Squeeze, The (1996)', 3)],
                     'incorrect movieNameWithAvgRatingsRDD.takeOrdered(3)')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


#### **(1c) Movies with Highest Average Ratings and more than 500 reviews**
#### Now that we have an RDD of the movies with highest averge ratings, we can use Spark to determine the 20 movies with highest average ratings and more than 500 reviews.
#### Apply a single RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the average rating to get the movies in order of their rating (highest rating first). You will end up with an RDD of the form: `[(4.5349264705882355, u'Shawshank Redemption, The (1994)', 1088), (4.515798462852263, u"Schindler's List (1993)", 1171), (4.512893982808023, u'Godfather, The (1972)', 1047)]`


    # TODO: Replace <FILL IN> with appropriate code
    
    # Apply an RDD transformation to `movieNameWithAvgRatingsRDD` to limit the results to movies with
    # ratings from more than 500 people. We then use the `sortFunction()` helper function to sort by the
    # average rating to get the movies in order of their rating (highest rating first)
    movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD.filter(lambda (avg, title, cnt): cnt > 500)
                                        .sortBy(sortFunction, False))
    print 'Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)

    Movies with highest ratings: [(4.5349264705882355, u'Shawshank Redemption, The (1994)', 1088), (4.515798462852263, u"Schindler's List (1993)", 1171), (4.512893982808023, u'Godfather, The (1972)', 1047), (4.510460251046025, u'Raiders of the Lost Ark (1981)', 1195), (4.505415162454874, u'Usual Suspects, The (1995)', 831), (4.457256461232604, u'Rear Window (1954)', 503), (4.45468509984639, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 651), (4.43953006219765, u'Star Wars: Episode IV - A New Hope (1977)', 1447), (4.4, u'Sixth Sense, The (1999)', 1110), (4.394285714285714, u'North by Northwest (1959)', 700), (4.379506641366224, u'Citizen Kane (1941)', 527), (4.375, u'Casablanca (1942)', 776), (4.363975155279503, u'Godfather: Part II, The (1974)', 805), (4.358816276202219, u"One Flew Over the Cuckoo's Nest (1975)", 811), (4.358173076923077, u'Silence of the Lambs, The (1991)', 1248), (4.335826477187734, u'Saving Private Ryan (1998)', 1337), (4.326241134751773, u'Chinatown (1974)', 564), (4.325383304940375, u'Life Is Beautiful (La Vita \ufffd bella) (1997)', 587), (4.324110671936759, u'Monty Python and the Holy Grail (1974)', 759), (4.3096, u'Matrix, The (1999)', 1250)]



    # TEST Movies with Highest Average Ratings and more than 500 Reviews (1c)
    
    Test.assertEquals(movieLimitedAndSortedByRatingRDD.count(), 194,
                    'incorrect movieLimitedAndSortedByRatingRDD.count()')
    Test.assertEquals(movieLimitedAndSortedByRatingRDD.take(20),
                  [(4.5349264705882355, u'Shawshank Redemption, The (1994)', 1088),
                   (4.515798462852263, u"Schindler's List (1993)", 1171),
                   (4.512893982808023, u'Godfather, The (1972)', 1047),
                   (4.510460251046025, u'Raiders of the Lost Ark (1981)', 1195),
                   (4.505415162454874, u'Usual Suspects, The (1995)', 831),
                   (4.457256461232604, u'Rear Window (1954)', 503),
                   (4.45468509984639, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 651),
                   (4.43953006219765, u'Star Wars: Episode IV - A New Hope (1977)', 1447),
                   (4.4, u'Sixth Sense, The (1999)', 1110), (4.394285714285714, u'North by Northwest (1959)', 700),
                   (4.379506641366224, u'Citizen Kane (1941)', 527), (4.375, u'Casablanca (1942)', 776),
                   (4.363975155279503, u'Godfather: Part II, The (1974)', 805),
                   (4.358816276202219, u"One Flew Over the Cuckoo's Nest (1975)", 811),
                   (4.358173076923077, u'Silence of the Lambs, The (1991)', 1248),
                   (4.335826477187734, u'Saving Private Ryan (1998)', 1337),
                   (4.326241134751773, u'Chinatown (1974)', 564),
                   (4.325383304940375, u'Life Is Beautiful (La Vita \ufffd bella) (1997)', 587),
                   (4.324110671936759, u'Monty Python and the Holy Grail (1974)', 759),
                   (4.3096, u'Matrix, The (1999)', 1250)], 'incorrect sortedByRatingRDD.take(20)')

    1 test passed.
    1 test passed.


#### Using a threshold on the number of reviews is one way to improve the recommendations, but there are many other good ways to improve quality. For example, you could weight ratings by the number of ratings.

## **Part 2: Collaborative Filtering**
#### In this course, you have learned about many of the basic transformations and actions that Spark allows us to apply to distributed datasets.  Spark also exposes some higher level functionality; in particular, Machine Learning using a component of Spark called [MLlib][mllib].  In this part, you will learn how to use MLlib to make personalized movie recommendations using the movie data we have been analyzing.
#### We are going to use a technique called [collaborative filtering][collab]. Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on an issue, A is more likely to have B's opinion on a different issue x than to have the opinion on x of a person chosen randomly. You can read more about collaborative filtering [here][collab2].
#### The image below (from [Wikipedia][collab]) shows an example of predicting of the user's rating using collaborative filtering. At first, people rate different items (like videos, images, games). After that, the system is making predictions about a user's rating for an item, which the user has not rated yet. These predictions are built upon the existing ratings of other users, who have similar ratings with the active user. For instance, in the image below the system has made a prediction, that the active user will not like the video.
![collaborative filtering](https://courses.edx.org/c4x/BerkeleyX/CS100.1x/asset/Collaborative_filtering.gif)
[mllib]: https://spark.apache.org/mllib/
[collab]: https://en.wikipedia.org/?title=Collaborative_filtering
[collab2]: http://recommender-systems.org/collaborative-filtering/

#### For movie recommendations, we start with a matrix whose entries are movie ratings by users (shown in red in the diagram below).  Each column represents a user (shown in green) and each row represents a particular movie (shown in blue).
#### Since not all users have rated all movies, we do not know all of the entries in this matrix, which is precisely why we need collaborative filtering.  For each user, we have ratings for only a subset of the movies.  With collaborative filtering, the idea is to approximate the ratings matrix by factorizing it as the product of two matrices: one that describes properties of each user (shown in green), and one that describes properties of each movie (shown in blue).
![factorization](http://spark-mooc.github.io/web-assets/images/matrix_factorization.png)
#### We want to select these two matrices such that the error for the users/movie pairs where we know the correct ratings is minimized.  The [Alternating Least Squares][als] algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the movies such that the error is minimized.  Then, it holds the movies matrix constrant and optimizes the value of the user's matrix.  This alternation between which matrix to optimize is the reason for the "alternating" in the name.
#### This optimization is what's being shown on the right in the image above.  Given a fixed set of user factors (i.e., values in the users matrix), we use the known ratings to find the best values for the movie factors using the optimization written at the bottom of the figure.  Then we "alternate" and pick the best user factors given fixed movie factors.
#### For a simple example of what the users and movies matrices might look like, check out the [videos from Lecture 8][videos] or the [slides from Lecture 8][slides]
[videos]: https://courses.edx.org/courses/BerkeleyX/CS100.1x/1T2015/courseware/00eb8b17939b4889a41a6d8d2f35db83/3bd3bba368be4102b40780550d3d8da6/
[slides]: https://courses.edx.org/c4x/BerkeleyX/CS100.1x/asset/Week4Lec8.pdf
[als]: https://en.wikiversity.org/wiki/Least-Squares_Method

#### **(2a) Creating a Training Set**
#### Before we jump into using machine learning, we need to break up the `ratingsRDD` dataset into three pieces:
* #### A training set (RDD), which we will use to train models
* #### A validation set (RDD), which we will use to choose the best model
* #### A test set (RDD), which we will use for our experiments
#### To randomly split the dataset into the multiple groups, we can use the pySpark [randomSplit()](https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.randomSplit) transformation. `randomSplit()` takes a set of splits and and seed and returns multiple RDDs.


    trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)
    
    print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                        validationRDD.count(),
                                                        testRDD.count())
    print trainingRDD.take(3)
    print validationRDD.take(3)
    print testRDD.take(3)
    
    assert trainingRDD.count() == 292716
    assert validationRDD.count() == 96902
    assert testRDD.count() == 98032
    
    assert trainingRDD.filter(lambda t: t == (1, 914, 3.0)).count() == 1
    assert trainingRDD.filter(lambda t: t == (1, 2355, 5.0)).count() == 1
    assert trainingRDD.filter(lambda t: t == (1, 595, 5.0)).count() == 1
    
    assert validationRDD.filter(lambda t: t == (1, 1287, 5.0)).count() == 1
    assert validationRDD.filter(lambda t: t == (1, 594, 4.0)).count() == 1
    assert validationRDD.filter(lambda t: t == (1, 1270, 5.0)).count() == 1
    
    assert testRDD.filter(lambda t: t == (1, 1193, 5.0)).count() == 1
    assert testRDD.filter(lambda t: t == (1, 2398, 4.0)).count() == 1
    assert testRDD.filter(lambda t: t == (1, 1035, 5.0)).count() == 1

    Training: 292716, validation: 96902, test: 98032
    
    [(1, 914, 3.0), (1, 2355, 5.0), (1, 595, 5.0)]
    [(1, 1287, 5.0), (1, 594, 4.0), (1, 1270, 5.0)]
    [(1, 1193, 5.0), (1, 2398, 4.0), (1, 1035, 5.0)]


#### After splitting the dataset, your training set has about 293,000 entries and the validation and test sets each have about 97,000 entries (the exact number of entries in each dataset varies slightly due to the random nature of the `randomSplit()` transformation.

#### **(2b) Root Mean Square Error (RMSE)**
#### In the next part, you will generate a few different models, and will need a way to decide which model is best. We will use the [Root Mean Square Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) or Root Mean Square Deviation (RMSD) to compute the error of each model.  RMSE is a frequently used measure of the differences between values (sample and population values) predicted by a model or an estimator and the values actually observed. The RMSD represents the sample standard deviation of the differences between predicted values and observed values. These individual differences are called residuals when the calculations are performed over the data sample that was used for estimation, and are called prediction errors when computed out-of-sample. The RMSE serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSE is a good measure of accuracy, but only to compare forecasting errors of different models for a particular variable and not between variables, as it is scale-dependent.
####  The RMSE is the square root of the average value of the square of `(actual rating - predicted rating)` for all users and movies for which we have the actual rating. Versions of Spark MLlib beginning with Spark 1.4 include a [RegressionMetrics](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RegressionMetrics) modiule that can be used to compute the RMSE. However, since we are using Spark 1.3.1, we will write our own function.
#### Write a function to compute the sum of squared error given `predictedRDD` and `actualRDD` RDDs. Both RDDs consist of tuples of the form (UserID, MovieID, Rating)
#### Given two ratings RDDs, *x* and *y* of size *n*, we define RSME as follows: $ RMSE = \sqrt{\frac{\sum_{i = 1}^{n} (x_i - y_i)^2}{n}}$
#### To calculate RSME, the steps you should perform are:
* #### Transform `predictedRDD` into the tuples of the form ((UserID, MovieID), Rating). For example, tuples like `[((1, 1), 5), ((1, 2), 3), ((1, 3), 4), ((2, 1), 3), ((2, 2), 2), ((2, 3), 4)]`. You can perform this step with a single Spark transformation.
* #### Transform `actualRDD` into the tuples of the form ((UserID, MovieID), Rating). For example, tuples like `[((1, 2), 3), ((1, 3), 5), ((2, 1), 5), ((2, 2), 1)]`. You can perform this step with a single Spark transformation.
* #### Using only RDD transformations (you only need to perform two transformations), compute the squared error for each *matching* entry (i.e., the same (UserID, MovieID) in each RDD) in the reformatted RDDs - do *not* use `collect()` to perform this step. Note that not every (UserID, MovieID) pair will appear in both RDDs - if a pair does not appear in both RDDs, then it does not contribute to the RMSE. You will end up with an RDD with entries of the form $ (x_i - y_i)^2$ You might want to check out Python's [math](https://docs.python.org/2/library/math.html) module to see how to compute these values
* #### Using an RDD action (but **not** `collect()`), compute the total squared error: $ SE = \sum_{i = 1}^{n} (x_i - y_i)^2 $
* #### Compute *n* by using an RDD action (but **not** `collect()`), to count the number of pairs for which you computed the total squared error
* #### Using the total squared error and the number of pairs, compute the RSME. Make sure you compute this value as a [float](https://docs.python.org/2/library/stdtypes.html#numeric-types-int-float-long-complex).
#### Note: Your solution must only use transformations and actions on RDDs. Do _not_ call `collect()` on either RDD.


    # TODO: Replace <FILL IN> with appropriate code
    import math
    
    def computeError(predictedRDD, actualRDD):
        """ Compute the root mean squared error between predicted and actual
        Args:
            predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                          (UserID, MovieID, Rating)
            actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
        Returns:
            RSME (float): computed RSME value
        """
        # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
        predictedReformattedRDD = predictedRDD.map(lambda (uid, mid, rat): ((uid, mid), rat))
    
        # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
        actualReformattedRDD = actualRDD.map(lambda (uid, mid, rat): ((uid, mid), rat))
    
        # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
        # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
        squaredErrorsRDD = (predictedReformattedRDD.join(actualReformattedRDD)
                            .map(lambda (k, rats): math.pow((rats[0] - rats[1]), 2)))
    
        # Compute the total squared error - do not use collect()
        totalError = squaredErrorsRDD.sum()
    
        # Count the number of entries for which you computed the total squared error
        numRatings = squaredErrorsRDD.count()
    
        # Using the total squared error and the number of entries, compute the RSME
        return math.sqrt(float(totalError)/numRatings)
    
    
    # sc.parallelize turns a Python list into a Spark RDD.
    testPredicted = sc.parallelize([
        (1, 1, 5),
        (1, 2, 3),
        (1, 3, 4),
        (2, 1, 3),
        (2, 2, 2),
        (2, 3, 4)])
    testActual = sc.parallelize([
         (1, 2, 3),
         (1, 3, 5),
         (2, 1, 5),
         (2, 2, 1)])
    testPredicted2 = sc.parallelize([
         (2, 2, 5),
         (1, 2, 5)])
    testError = computeError(testPredicted, testActual)
    print 'Error for test dataset (should be 1.22474487139): %s' % testError
    
    testError2 = computeError(testPredicted2, testActual)
    print 'Error for test dataset2 (should be 3.16227766017): %s' % testError2
    
    testError3 = computeError(testActual, testActual)
    print 'Error for testActual dataset (should be 0.0): %s' % testError3

    Error for test dataset (should be 1.22474487139): 1.22474487139
    Error for test dataset2 (should be 3.16227766017): 3.16227766017
    Error for testActual dataset (should be 0.0): 0.0



    # TEST Root Mean Square Error (2b)
    Test.assertTrue(abs(testError - 1.22474487139) < 0.00000001,
                    'incorrect testError (expected 1.22474487139)')
    Test.assertTrue(abs(testError2 - 3.16227766017) < 0.00000001,
                    'incorrect testError2 result (expected 3.16227766017)')
    Test.assertTrue(abs(testError3 - 0.0) < 0.00000001,
                    'incorrect testActual result (expected 0.0)')

    1 test passed.
    1 test passed.
    1 test passed.


#### **(2c) Using ALS.train()**
#### In this part, we will use the MLlib implementation of Alternating Least Squares, [ALS.train()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS). ALS takes a training dataset (RDD) and several parameters that control the model creation process. To determine the best values for the parameters, we will use ALS to train several models, and then we will select the best model and use the parameters from that model in the rest of this lab exercise.
#### The process we will use for determining the best model is as follows:
* #### Pick a set of model parameters. The most important parameter to `ALS.train()` is the *rank*, which is the number of rows in the Users matrix (green in the diagram above) or the number of columns in the Movies matrix (blue in the diagram above). (In general, a lower rank will mean higher error on the training dataset, but a high rank may lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting).)  We will train models with ranks of 4, 8, and 12 using the `trainingRDD` dataset.
* #### Create a model using `ALS.train(trainingRDD, rank, seed=seed, iterations=iterations, lambda_=regularizationParameter)` with three parameters: an RDD consisting of tuples of the form (UserID, MovieID, rating) used to train the model, an integer rank (4, 8, or 12), a number of iterations to execute (we will use 5 for the `iterations` parameter), and a regularization coefficient (we will use 0.1 for the `regularizationParameter`).
* #### For the prediction step, create an input RDD, `validationForPredictRDD`, consisting of (UserID, MovieID) pairs that you extract from `validationRDD`. You will end up with an RDD of the form: `[(1, 1287), (1, 594), (1, 1270)]`
* #### Using the model and `validationForPredictRDD`, we can predict rating values by calling [model.predictAll()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.predictAll) with the `validationForPredictRDD` dataset, where `model` is the model we generated with ALS.train().  `predictAll` accepts an RDD with each entry in the format (userID, movieID) and outputs an RDD with each entry in the format (userID, movieID, rating).
* #### Evaluate the quality of the model by using the `computeError()` function you wrote in part (2b) to compute the error between the predicted ratings and the actual ratings in `validationRDD`.
####  Which rank produces the best model, based on the RMSE with the `validationRDD` dataset?
#### Note: It is likely that this operation will take a noticeable amount of time (around a minute in our VM); you can observe its progress on the [Spark Web UI](http://localhost:4040). Probably most of the time will be spent running your `computeError()` function, since, unlike the Spark ALS implementation (and the Spark 1.4 [RegressionMetrics](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RegressionMetrics) module), this does not use a fast linear algebra library and needs to run some Python code for all 100k entries.


    # TODO: Replace <FILL IN> with appropriate code
    from pyspark.mllib.recommendation import ALS
    
    validationForPredictRDD = validationRDD.map(lambda (uid, mid, rat): (uid, mid))
    
    seed = 5L
    iterations = 5
    regularizationParameter = 0.1
    ranks = [x for x in range(1, 21)]
    errors = [0 for x in ranks]
    err = 0
    tolerance = 0.02
    
    minError = float('inf')
    bestRank = -1
    bestIteration = -1
    for rank in ranks:
        model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularizationParameter)
        predictedRatingsRDD = model.predictAll(validationForPredictRDD)
        error = computeError(predictedRatingsRDD, validationRDD)
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < minError:
            minError = error
            bestRank = rank
    
    print 'The best model was trained with rank %s' % bestRank

    For rank 1 the RMSE is 0.920257595191
    For rank 2 the RMSE is 0.900974642908
    For rank 3 the RMSE is 0.903389356915
    For rank 4 the RMSE is 0.892734779484
    For rank 5 the RMSE is 0.886917879277
    For rank 6 the RMSE is 0.894298306606
    For rank 7 the RMSE is 0.895577476117
    For rank 8 the RMSE is 0.890121292255
    For rank 9 the RMSE is 0.890636803379
    For rank 10 the RMSE is 0.893253335162
    For rank 11 the RMSE is 0.884786415609
    For rank 12 the RMSE is 0.890216118367
    For rank 13 the RMSE is 0.893286091627
    For rank 14 the RMSE is 0.893915622604
    For rank 15 the RMSE is 0.894303209046
    For rank 16 the RMSE is 0.89246441948
    For rank 17 the RMSE is 0.891385245908
    For rank 18 the RMSE is 0.893526682163
    For rank 19 the RMSE is 0.892553471967
    For rank 20 the RMSE is 0.895563571352
    The best model was trained with rank 11



    # TEST Using ALS.train (2c)
    Test.assertEquals(trainingRDD.getNumPartitions(), 2,
                      'incorrect number of partitions for trainingRDD (expected 2)')
    Test.assertEquals(validationForPredictRDD.count(), 96902,
                      'incorrect size for validationForPredictRDD (expected 96902)')
    Test.assertEquals(validationForPredictRDD.filter(lambda t: t == (1, 1907)).count(), 1,
                      'incorrect content for validationForPredictRDD')
    Test.assertTrue(abs(errors[0] - 0.883710109497) < tolerance, 'incorrect errors[0]')
    Test.assertTrue(abs(errors[1] - 0.878486305621) < tolerance, 'incorrect errors[1]')
    Test.assertTrue(abs(errors[2] - 0.876832795659) < tolerance, 'incorrect errors[2]')

    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.
    1 test passed.


#### **(2d) Testing Your Model**
#### So far, we used the `trainingRDD` and `validationRDD` datasets to select the best model.  Since we used these two datasets to determine what model is best, we cannot use them to test how good the model is - otherwise we would be very vulnerable to [overfitting](https://en.wikipedia.org/wiki/Overfitting).  To decide how good our model is, we need to use the `testRDD` dataset.  We will use the `bestRank` you determined in part (2c) to create a model for predicting the ratings for the test dataset and then we will compute the RMSE.
#### The steps you should perform are:
* #### Train a model, using the `trainingRDD`, `bestRank` from part (2c), and the parameters you used in in part (2c): `seed=seed`, `iterations=iterations`, and `lambda_=regularizationParameter` - make sure you include **all** of the parameters.
* #### For the prediction step, create an input RDD, `testForPredictingRDD`, consisting of (UserID, MovieID) pairs that you extract from `testRDD`. You will end up with an RDD of the form: `[(1, 1287), (1, 594), (1, 1270)]`
* #### Use [myModel.predictAll()](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.predictAll) to predict rating values for the test dataset.
* #### For validation, use the `testRDD`and your `computeError` function to compute the RMSE between `testRDD` and the `predictedTestRDD` from the model.
* #### Evaluate the quality of the model by using the `computeError()` function you wrote in part (2b) to compute the error between the predicted ratings and the actual ratings in `testRDD`.


    # TODO: Replace <FILL IN> with appropriate code
    rank = bestRank
    seed = 5L
    iterations = 5
    regularizationParameter = 0.1
    
    myModel = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations, lambda_=regularizationParameter)
    testForPredictingRDD = testRDD.map(lambda (uid, mid, rat): (uid, mid))
    predictedTestRDD = myModel.predictAll(testForPredictingRDD)
    
    testRMSE = computeError(predictedTestRDD, testRDD)
    
    print 'The model had a RMSE on the test set of %s' % testRMSE

    The model had a RMSE on the test set of 0.886682716205



    # TEST Testing Your Model (2d)
    Test.assertTrue(abs(testRMSE - 0.87809838344) < tolerance, 'incorrect testRMSE')

    1 test passed.


#### **(2e) Comparing Your Model**
#### Looking at the RMSE for the results predicted by the model versus the values in the test set is one way to evalute the quality of our model. Another way to evaluate the model is to evaluate the error from a test set where every rating is the average rating for the training set.
#### The steps you should perform are:
* #### Use the `trainingRDD` to compute the average rating across all movies in that training dataset.
* #### Use the average rating that you just determined and the `testRDD` to create an RDD with entries of the form (userID, movieID, average rating).
* #### Use your `computeError` function to compute the RMSE between the `testRDD` validation RDD that you just created and the `testForAvgRDD`.


    # TODO: Replace <FILL IN> with appropriate code
    
    trainingAvgRating = trainingRDD.map(lambda (uid, mid, rat): rat).mean()
    print 'The average rating for movies in the training set is %s' % trainingAvgRating
    
    testForAvgRDD = testRDD.map(lambda (uid, mid, rat): (uid, mid, trainingAvgRating))
    testAvgRMSE = computeError(testForAvgRDD, testRDD)
    print 'The RMSE on the average set is %s' % testAvgRMSE

    The average rating for movies in the training set is 3.57409571052
    The RMSE on the average set is 1.12036693569



    # TEST Comparing Your Model (2e)
    Test.assertTrue(abs(trainingAvgRating - 3.57409571052) < 0.000001,
                    'incorrect trainingAvgRating (expected 3.57409571052)')
    Test.assertTrue(abs(testAvgRMSE - 1.12036693569) < 0.000001,
                    'incorrect testAvgRMSE (expected 1.12036693569)')

    1 test passed.
    1 test passed.


#### You now have code to predict how users will rate movies!

## **Part 3: Predictions for Yourself**
#### The ultimate goal of this lab exercise is to predict what movies to recommend to yourself.  In order to do that, you will first need to add ratings for yourself to the `ratingsRDD` dataset.

#### **(3a) Your Movie Ratings**
#### To help you provide ratings for yourself, we have included the following code to list the names and movie IDs of the 50 highest-rated movies from `movieLimitedAndSortedByRatingRDD` which we created in part 1 the lab.


    moviesSortedRDD = (moviesRDD.join(movieIDsWithAvgRatingsRDD).map(lambda (mid, (title, (cnt, avg))): (mid, avg, title, cnt))
                       .filter(lambda (mid, avg, title, cnt): cnt > 500)
                        .sortBy(lambda (mid, avg, title, cnt): avg, False))
    print 'Most rated movies:'
    print '(movie id, average rating, movie name, number of reviews)'
    for ratingsTuple in moviesSortedRDD.take(73):
        print ratingsTuple

    Most rated movies:
    (movie id, average rating, movie name, number of reviews)
    (318, 4.5349264705882355, u'Shawshank Redemption, The (1994)', 1088)
    (527, 4.515798462852263, u"Schindler's List (1993)", 1171)
    (858, 4.512893982808023, u'Godfather, The (1972)', 1047)
    (1198, 4.510460251046025, u'Raiders of the Lost Ark (1981)', 1195)
    (50, 4.505415162454874, u'Usual Suspects, The (1995)', 831)
    (904, 4.457256461232604, u'Rear Window (1954)', 503)
    (750, 4.45468509984639, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 651)
    (260, 4.43953006219765, u'Star Wars: Episode IV - A New Hope (1977)', 1447)
    (2762, 4.4, u'Sixth Sense, The (1999)', 1110)
    (908, 4.394285714285714, u'North by Northwest (1959)', 700)
    (923, 4.379506641366224, u'Citizen Kane (1941)', 527)
    (912, 4.375, u'Casablanca (1942)', 776)
    (1221, 4.363975155279503, u'Godfather: Part II, The (1974)', 805)
    (1193, 4.358816276202219, u"One Flew Over the Cuckoo's Nest (1975)", 811)
    (593, 4.358173076923077, u'Silence of the Lambs, The (1991)', 1248)
    (2028, 4.335826477187734, u'Saving Private Ryan (1998)', 1337)
    (1252, 4.326241134751773, u'Chinatown (1974)', 564)
    (2324, 4.325383304940375, u'Life Is Beautiful (La Vita \ufffd bella) (1997)', 587)
    (1136, 4.324110671936759, u'Monty Python and the Holy Grail (1974)', 759)
    (2571, 4.3096, u'Matrix, The (1999)', 1250)
    (1196, 4.309457579972183, u'Star Wars: Episode V - The Empire Strikes Back (1980)', 1438)
    (1278, 4.30379746835443, u'Young Frankenstein (1974)', 553)
    (1219, 4.301346801346801, u'Psycho (1960)', 594)
    (296, 4.296438883541867, u'Pulp Fiction (1994)', 1039)
    (608, 4.286535303776683, u'Fargo (1996)', 1218)
    (1213, 4.282367447595561, u'GoodFellas (1990)', 811)
    (2858, 4.27943661971831, u'American Beauty (1999)', 1775)
    (919, 4.268053855569155, u'Wizard of Oz, The (1939)', 817)
    (1197, 4.267774699907664, u'Princess Bride, The (1987)', 1083)
    (1247, 4.253333333333333, u'Graduate, The (1967)', 600)
    (2692, 4.236263736263736, u'Run Lola Run (Lola rennt) (1998)', 546)
    (1225, 4.233807266982622, u'Amadeus (1984)', 633)
    (3114, 4.232558139534884, u'Toy Story 2 (1999)', 860)
    (1288, 4.232558139534884, u'This Is Spinal Tap (1984)', 516)
    (3897, 4.228494623655914, u'Almost Famous (2000)', 744)
    (2804, 4.2250755287009065, u'Christmas Story, A (1983)', 662)
    (1242, 4.216757741347905, u'Glory (1989)', 549)
    (1208, 4.213358070500927, u'Apocalypse Now (1979)', 539)
    (1617, 4.20992028343667, u'L.A. Confidential (1997)', 1129)
    (541, 4.204733727810651, u'Blade Runner (1982)', 845)
    (1358, 4.1886120996441285, u'Sling Blade (1996)', 562)
    (110, 4.184615384615385, u'Braveheart (1995)', 1300)
    (1304, 4.184168012924071, u'Butch Cassidy and the Sundance Kid (1969)', 619)
    (1704, 4.182509505703422, u'Good Will Hunting (1997)', 789)
    (111, 4.166969147005445, u'Taxi Driver (1976)', 551)
    (1240, 4.162767039674466, u'Terminator, The (1984)', 983)
    (1089, 4.157545605306799, u'Reservoir Dogs (1992)', 603)
    (1387, 4.153333333333333, u'Jaws (1975)', 750)
    (1214, 4.149840595111583, u'Alien (1979)', 941)
    (1, 4.145015105740181, u'Toy Story (1995)', 993)
    (1294, 4.142857142857143, u'M*A*S*H (1970)', 518)
    (2918, 4.129737609329446, u"Ferris Bueller's Day Off (1986)", 686)
    (1036, 4.124678663239075, u'Die Hard (1988)', 778)
    (1200, 4.122596153846154, u'Aliens (1986)', 832)
    (356, 4.121270452358036, u'Forrest Gump (1994)', 1039)
    (1291, 4.11251580278129, u'Indiana Jones and the Last Crusade (1989)', 791)
    (1230, 4.111470113085622, u'Annie Hall (1977)', 619)
    (3147, 4.107407407407408, u'Green Mile, The (1999)', 540)
    (2396, 4.092905405405405, u'Shakespeare in Love (1998)', 1184)
    (1222, 4.090289608177172, u'Full Metal Jacket (1987)', 587)
    (150, 4.084126984126984, u'Apollo 13 (1995)', 630)
    (2997, 4.083788706739527, u'Being John Malkovich (1999)', 1098)
    (1259, 4.081309398099261, u'Stand by Me (1986)', 947)
    (924, 4.07521578298397, u'2001: A Space Odyssey (1968)', 811)
    (589, 4.075182481751825, u'Terminator 2: Judgment Day (1991)', 1370)
    (1206, 4.075043630017452, u'Clockwork Orange, A (1971)', 573)
    (1610, 4.072258064516129, u'Hunt for Red October, The (1990)', 775)
    (457, 4.071129707112971, u'Fugitive, The (1993)', 956)
    (1961, 4.068078668683812, u'Rain Man (1988)', 661)
    (1090, 4.057971014492754, u'Platoon (1986)', 552)
    (920, 4.053763440860215, u'Gone with the Wind (1939)', 558)
    (1307, 4.051724137931035, u'When Harry Met Sally... (1989)', 754)
    (1394, 4.050151975683891, u'Raising Arizona (1987)', 658)


#### The user ID 0 is unassigned, so we will use it for your ratings. We set the variable `myUserID` to 0 for you. Next, create a new RDD `myRatingsRDD` with your ratings for at least 10 movie ratings. Each entry should be formatted as `(myUserID, movieID, rating)` (i.e., each entry should be formatted in the same way as `trainingRDD`).  As in the original dataset, ratings should be between 1 and 5 (inclusive). If you have not seen at least 10 of these movies, you can increase the parameter passed to `take()` in the above cell until there are 10 movies that you have seen (or you can also guess what your rating would be for movies you have not seen).


    # TODO: Replace <FILL IN> with appropriate code
    myUserID = 0
    
    # Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
    myRatedMovies = [(myUserID, 318, 5), (myUserID, 527, 5), (myUserID, 858, 4), (myUserID, 1198, 4), (myUserID, 50, 5),
                     (myUserID, 260, 4), (myUserID, 2762, 5), (myUserID, 1193, 4), (myUserID, 593, 5), (myUserID, 2324, 5),
                     (myUserID, 1136, 5), (myUserID, 2571, 5), (myUserID, 1196, 4), (myUserID, 608, 5), (myUserID, 1213, 5),
                     (myUserID, 2858, 4), (myUserID, 2692, 5), (myUserID, 3114, 5), (myUserID, 356, 5), (myUserID, 3147, 5)]
    myRatingsRDD = sc.parallelize(myRatedMovies)
    print 'My movie ratings: %s' % myRatingsRDD.take(10)

    My movie ratings: [(0, 318, 5), (0, 527, 5), (0, 858, 4), (0, 1198, 4), (0, 50, 5), (0, 260, 4), (0, 2762, 5), (0, 1193, 4), (0, 593, 5), (0, 2324, 5)]


#### **(3b) Add Your Movies to Training Dataset**
#### Now that you have ratings for yourself, you need to add your ratings to the `training` dataset so that the model you train will incorporate your preferences.  Spark's [union()](http://spark.apache.org/docs/latest/api/python/pyspark.rdd.RDD-class.html#union) transformation combines two RDDs; use `union()` to create a new training dataset that includes your ratings and the data in the original training dataset.


    # TODO: Replace <FILL IN> with appropriate code
    trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)
    
    print ('The training dataset now has %s more entries than the original training dataset' %
           (trainingWithMyRatingsRDD.count() - trainingRDD.count()))
    assert (trainingWithMyRatingsRDD.count() - trainingRDD.count()) == myRatingsRDD.count()

    The training dataset now has 20 more entries than the original training dataset


#### **(3c) Train a Model with Your Ratings**
#### Now, train a model with your ratings added and the parameters you used in in part (2c): `bestRank`, `seed=seed`, `iterations=iterations`, and `lambda_=regularizationParameter` - make sure you include **all** of the parameters.


    # TODO: Replace <FILL IN> with appropriate code
    myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,
                               lambda_=regularizationParameter)

#### **(3d) Check RMSE for the New Model with Your Ratings**
#### Compute the RMSE for this new model on the test set.
* #### For the prediction step, we reuse `testForPredictingRDD`, consisting of (UserID, MovieID) pairs that you extracted from `testRDD`. The RDD has the form: `[(1, 1287), (1, 594), (1, 1270)]`
* #### Use `myRatingsModel.predictAll()` to predict rating values for the `testForPredictingRDD` test dataset, set this as `predictedTestMyRatingsRDD`
* #### For validation, use the `testRDD`and your `computeError` function to compute the RMSE between `testRDD` and the `predictedTestMyRatingsRDD` from the model.


    # TODO: Replace <FILL IN> with appropriate code
    predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
    testRMSEMyRatings = computeError(predictedTestMyRatingsRDD, testRDD)
    print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings

    The model had a RMSE on the test set of 0.89187453878


#### **(3e) Predict Your Ratings**
#### So far, we have only used the `predictAll` method to compute the error of the model.  Here, use the `predictAll` to predict what ratings you would give to the movies that you did not already provide ratings for.
#### The steps you should perform are:
* #### Use the Python list `myRatedMovies` to transform the `moviesRDD` into an RDD with entries that are pairs of the form (myUserID, Movie ID) and that does not contain any movies that you have rated. This transformation will yield an RDD of the form: `[(0, 1), (0, 2), (0, 3), (0, 4)]`. Note that you can do this step with one RDD transformation.
* #### For the prediction step, use the input RDD, `myUnratedMoviesRDD`, with myRatingsModel.predictAll() to predict your ratings for the movies.


    # TODO: Replace <FILL IN> with appropriate code
    
    # Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of the form (myUserID, Movie ID) and that does not contain any movies that you have rated.
    myUnratedMoviesRDD = (moviesRDD.leftOuterJoin(myRatingsRDD.map(lambda (uid, mid, rat): (mid, (uid, rat))))
                             .filter(lambda (mid, (title, ratDet)): ratDet == None)
                              .map(lambda (mid, (title, ratDet)): (myUserID, mid)))
    
    # Use the input RDD, myUnratedMoviesRDD, with myRatingsModel.predictAll() to predict your ratings for the movies
    predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)

#### **(3f) Predict Your Ratings**
#### We have our predicted ratings. Now we can print out the 25 movies with the highest predicted ratings.
#### The steps you should perform are:
* #### From Parts (1b) and (1c), we know that we should look at movies with a reasonable number of reviews (e.g., more than 75 reviews). You can experiment with a lower threshold, but fewer ratings for a movie may yield higher prediction errors. Transform `movieIDsWithAvgRatingsRDD` from Part (1b), which has the form (MovieID, (number of ratings, average rating)), into an RDD of the form (MovieID, number of ratings): `[(2, 332), (4, 71), (6, 442)]`
* #### We want to see movie names, instead of movie IDs. Transform `predictedRatingsRDD` into an RDD with entries that are pairs of the form (Movie ID, Predicted Rating): `[(3456, -0.5501005376936687), (1080, 1.5885892024487962), (320, -3.7952255522487865)]`
* #### Use RDD transformations with `predictedRDD` and `movieCountsRDD` to yield an RDD with tuples of the form (Movie ID, (Predicted Rating, number of ratings)): `[(2050, (0.6694097486155939, 44)), (10, (5.29762541533513, 418)), (2060, (0.5055259373841172, 97))]`
* #### Use RDD transformations with `predictedWithCountsRDD` and `moviesRDD` to yield an RDD with tuples of the form (Predicted Rating, Movie Name, number of ratings), _for movies with more than 75 ratings._ For example: `[(7.983121900375243, u'Under Siege (1992)'), (7.9769201864261285, u'Fifth Element, The (1997)')]`


    # TODO: Replace <FILL IN> with appropriate code
    
    # Transform movieIDsWithAvgRatingsRDD from part (1b), which has the form (MovieID, (number of ratings, average rating)), into and RDD of the form (MovieID, number of ratings)
    #movieCountsRDD = movieIDsWithAvgRatingsRDD.<FILL IN>
    
    # Transform predictedRatingsRDD into an RDD with entries that are pairs of the form (Movie ID, Predicted Rating)
    predictedRDD = predictedRatingsRDD.map(lambda (uid, mid, predRat): (mid, predRat))
    
    # Use RDD transformations with predictedRDD and movieCountsRDD to yield an RDD with tuples of the form (Movie ID, (Predicted Rating, number of ratings))
    #predictedWithCountsRDD  = (predictedRDD.<FILL IN>)
    
    # Use RDD transformations with PredictedWithCountsRDD and moviesRDD to yield an RDD with tuples of the form (Predicted Rating, Movie Name, number of ratings), for movies with more than 75 ratings
    #ratingsWithNamesRDD = (predictedWithCountsRDD.<FILL IN>)
    
    moviesToCheckPred = (moviesRDD.join(movieIDsWithAvgRatingsRDD).map(lambda (mid, (title, (cnt, avg))): (mid, (title, avg, cnt)))
                         .filter(lambda (mid, (title, avg, cnt)): cnt > 75))
    
    ratingsWithNamesRDD = (moviesToCheckPred.join(predictedRDD)
                           .map( lambda (mid, ((title, avg, cnt), predRat)): (mid, title, predRat, avg, cnt) )
                          )
    
    predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[2])
    print 'My highest rated movies as predicted (for movies with more than 75 reviews):'
    print ('Movie ID, Movie Name, Predicted Rating, Average User Rating, Rating Count\n%s' %
            '\n'.join(map(str, predictedHighestRatedMovies)))

    My highest rated movies as predicted (for movies with more than 75 reviews):
    Movie ID, Movie Name, Predicted Rating, Average User Rating, Rating Count
    (745, u'Close Shave, A (1995)', 4.740393433277688, 4.490566037735849, 318)
    (1148, u'Wrong Trousers, The (1993)', 4.729642824990393, 4.531764705882353, 425)
    (2019, u'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)', 4.648314421847047, 4.586330935251799, 278)
    (1207, u'To Kill a Mockingbird (1962)', 4.607289507499839, 4.450110864745011, 451)
    (1131, u'Jean de Florette (1986)', 4.597714104735545, 4.397849462365591, 93)
    (913, u'Maltese Falcon, The (1941)', 4.577007197436432, 4.388655462184874, 476)
    (1250, u'Bridge on the River Kwai, The (1957)', 4.568229336916544, 4.390852390852391, 481)
    (1178, u'Paths of Glory (1957)', 4.550100326692569, 4.571428571428571, 105)
    (2804, u'Christmas Story, A (1983)', 4.547580509895146, 4.2250755287009065, 662)
    (954, u'Mr. Smith Goes to Washington (1939)', 4.547210358641717, 4.251461988304094, 171)
    (2028, u'Saving Private Ryan (1998)', 4.547186563376663, 4.335826477187734, 1337)
    (720, u'Wallace & Gromit: The Best of Aardman Animation (1996)', 4.546019161087186, 4.465346534653466, 202)
    (1704, u'Good Will Hunting (1997)', 4.539002853016079, 4.182509505703422, 789)
    (2357, u'Central Station (Central do Brasil) (1998)', 4.532924861082282, 4.300970873786408, 103)
    (1262, u'Great Escape, The (1963)', 4.525140602688774, 4.35, 360)
    (904, u'Rear Window (1954)', 4.52336468419648, 4.457256461232604, 503)
    (1299, u'Killing Fields, The (1984)', 4.523031354492897, 4.289017341040463, 346)
    (3429, u'Creature Comforts (1990)', 4.518349318806957, 4.345323741007194, 139)
    (908, u'North by Northwest (1959)', 4.514540848589619, 4.394285714285714, 700)
    (1203, u'12 Angry Men (1957)', 4.51260151769687, 4.3152866242038215, 314)



    # Get all code together to see how predictions change as more ratings are provided
    # Ideas (or hypothesis, if you will) to test:
    # 1. High correlation of your strong ratings (very high or very low rating) with population produces predicted ratings
    #    close to population average
    #      a. You provided high ratings to 20 movies from the top50 list (hyp validation)
    #      b. You provided low ratings to 20 movies from the bottom50 list (hyp validation)
    #      c. You provided a mix (10 each) from (a) and (b) (hyp validation)
    #      d. You provided low ratings to 20 of the top50 movies (hyp challenge)
    #      e. You provided high ratings to 20 of the bottom50 movies (hyp challenge)
    #      f. You provided a mix (10 each) from (d) and (e) (hyp challenge)
    #      g. You provided high ratings to 10 movies near the mean index and low ratings to the next 10 movies (hyp challenge)
    
    
    import math
    import random
    ########################## global parameters # Start
    # keep same count cutoff for providing ratings as well as recommending? same value ensures same movies population...
    minRatedCount = 200
    minCountForReco = 200
    countRecoMovies = 20
    # variables set to lab's choices, but you can change to play around; rank is best left unchanged
    modelRank = bestRank
    modelSeed = seed
    modelLoops = iterations
    modelLambda = regularizationParameter
    # switch to use manual entry for rating movies
    # or auto rating based on rank - top 10 rated 5, next 10 rated 4, bottom 10 rated 1, previous 10 from bottom rated 2
    useManualRatings = 1
    # create list (a list of lists) to store your movie ratings and loop over for each hyp, especially if manual switch = 0
    hypNames = []
    hypSelectRanks = []
    hypAssignRatings = []
    hypRatedMovieDetails = []
    hypRatings = []
    hypPredictions = []
    # set the number of movies to provide auto-ratings for
    reviewMoviesCount = 20
    # should we randomly select 'reviewMoviesCount' movies from (randomSelectionBase * reviewMoviesCount) movies?
    # use 1 to use only the top / bottom reviewMoviesCount movies, or > 1 to get a random sampling
    randomSelectionBase = 2
    
    # variables used from the lab: moviesRDD, movieIDsWithAvgRatingsRDD, trainingRDD, testForPredictingRDD, testRDD
    # all other RDDs are created and used in this cell
    ########################## global parameters # End
    
    
    movieDetailsRDD = (moviesRDD.join(movieIDsWithAvgRatingsRDD)
                       .map(lambda (mid, (title, (cnt, avg))): (mid, (title, avg, cnt))).cache())
    
    # this if-block can be ported out as-is to a function; will help modularity, reduce code clutter
    # signature of this function will be (IN: useManualRatings, OUT: hypNames, hypRatings, hypRatedMovieDetails)
    if useManualRatings:
    #    print 'Most rated movies:'
    #    print '(movie id, average rating, movie name, number of reviews)'
    #    for ratingsTuple in (movieDetailsRDD.filter(lambda (mid, (title, avg, cnt)): cnt > minRatedCount)
    #                             .takeOrdered(73, key = lambda (mid, (title, avg, cnt)): -1 * avg)):
    #        print ratingsTuple
        myMovies = [(0, 318, 5), (0, 527, 5), (0, 858, 4), (0, 1198, 5), (0, 50, 5), (0, 260, 4), (0, 2762, 5), (0, 1193, 4),
                    (0, 593, 5), (0, 2324, 5), (0, 1136, 5), (0, 2571, 5), (0, 1196, 4), (0, 608, 5), (0, 1213, 5),
                    (0, 2858, 5), (0, 2692, 5), (0, 3114, 5), (0, 356, 5), (0, 3147, 5)]
        hypNames.append('Manual Rating of Movies')
        hypRatings.append(myMovies)
        
        moviesSortedRDD = movieDetailsRDD.filter(lambda (mid, (title, avg, cnt)): cnt > minRatedCount).sortByKey()
        ratedDetails = []
        for movie in myMovies:
            movDetails = moviesSortedRDD.lookup(movie[1])
            ratedDetails.append( (movie[1], (movDetails[0][0], movDetails[0][1], movDetails[0][2], movie[2])) )
        hypRatedMovieDetails.append(ratedDetails)
        movDetails = []
        ratedDetails = []
        moviesSortedRDD.unpersist()
    else:
        # initialize lists to auto-generate the ratings
        hypNames = ['Hyp 1: High Rating to top rated movies', 'Hyp 2: Low Rating to lowest rated movies',
                    'Hyp 3: Mix of High Rating to top and Low Rating to bottom movies',
                    'Hyp 4: Low Rating to top rated movies', 'Hyp 5: High Rating to lowest rated movies',
                    'Hyp 6: Mix of Low Rating to top and High Rating to bottom movies']
        hypSelectRanks = [(reviewMoviesCount, 0), (0, reviewMoviesCount), (reviewMoviesCount/2, reviewMoviesCount/2),
                          (reviewMoviesCount, 0), (0, reviewMoviesCount), (reviewMoviesCount/2, reviewMoviesCount/2)]
        hypAssignRatings = [(5, 1), (5, 1), (5, 1), (1, 5), (1, 5), (1, 5)]
        
        for idx in range(len(hypNames)):
            topTake = hypSelectRanks[idx][0]
            botTake = hypSelectRanks[idx][1]
            topRating = hypAssignRatings[idx][0]
            botRating = hypAssignRatings[idx][1]
            
            tops = random.sample((movieDetailsRDD.filter(lambda (mid, (title, avg, cnt)): cnt > minRatedCount)
                                    .takeOrdered(randomSelectionBase * topTake, key = lambda (mid, (title, avg, cnt)): -1*avg)),
                                 topTake)
            bots = random.sample((movieDetailsRDD.filter(lambda (mid, (title, avg, cnt)): cnt > minRatedCount)
                                    .takeOrdered(randomSelectionBase * botTake, key = lambda (mid, (title, avg, cnt)): avg)),
                                 botTake)
            
            hypRatings.append([(0, x[0], topRating) for x in tops] + [(0, x[0], botRating) for x in bots])
            hypRatedMovieDetails.append([(x[0], (x[1][0], x[1][1], x[1][2], topRating)) for x in tops] +
                                        [(x[0], (x[1][0], x[1][1], x[1][2], botRating)) for x in bots])
        
        tops = []
        bots = []
    
    
    for idx, moviesRated in enumerate(hypRatings):
        print('\n%s - Movies Rated by you' % hypNames[idx])
        print('(Movie ID, (Movie Title, Avg User Rating, Ratings Count, Your Rating))\n%s' %
                 '\n'.join(map(str, hypRatedMovieDetails[idx])))
        
        myMoviesRDD = sc.parallelize(moviesRated)
        trainingMyMoviesRDD = trainingRDD.union(myMoviesRDD)
    
        myMoviesModel = ALS.train(trainingMyMoviesRDD, modelRank, seed=modelSeed, iterations=modelLoops, lambda_=modelLambda)
    
        predTestMyMoviesRDD = myMoviesModel.predictAll(testForPredictingRDD)
        testRMSEMyMovies = computeError(predTestMyMoviesRDD, testRDD)
        print('\nLatent factors used to predict user ratings = %s' % modelRank)
        print('Model RMSE on the test set = %s' % testRMSEMyMovies)
    
        notWatchedMoviesRDD = (moviesRDD.leftOuterJoin(myMoviesRDD.map(lambda (uid, mid, rat): (mid, (uid, rat))))
                                 .filter(lambda (mid, (title, ratDet)): ratDet == None)
                                  .map(lambda (mid, (title, ratDet)): (0, mid)))
        predRatingsRDD = myMoviesModel.predictAll(notWatchedMoviesRDD)
    
        predRDD = predRatingsRDD.map(lambda (uid, mid, predRat): (mid, predRat))
        targetMoviesPred = movieDetailsRDD.filter(lambda (mid, (title, avg, cnt)): cnt > minCountForReco)
        targetMoviesRatingsRDD = (targetMoviesPred.join(predRDD)
                                    .map( lambda (mid, ((title, avg, cnt), predRat)): (mid, title, predRat, avg, cnt) ))
    
        predHighestRatedNotWatchedMovies = targetMoviesRatingsRDD.takeOrdered(countRecoMovies, key=lambda x: -x[2])
        hypPredictions.append(predHighestRatedNotWatchedMovies)
        print('\nYour top-rated %s movies as predicted (for movies with more than %s reviews):' %(countRecoMovies, minCountForReco))
        print('Movie ID, Movie Name, Predicted Rating, Average User Rating, Rating Count')
        for printIdx, predMovie in enumerate(predHighestRatedNotWatchedMovies, start = 1):
            print ('%s. %s' % (printIdx, predMovie))
    
    print('\n\n************* Hypothesis testing on Recommender System completed *************')

    
    Manual Rating of Movies - Movies Rated by you
    (Movie ID, (Movie Title, Avg User Rating, Ratings Count, Your Rating))
    (318, (u'Shawshank Redemption, The (1994)', 4.5349264705882355, 1088, 5))
    (527, (u"Schindler's List (1993)", 4.515798462852263, 1171, 5))
    (858, (u'Godfather, The (1972)', 4.512893982808023, 1047, 4))
    (1198, (u'Raiders of the Lost Ark (1981)', 4.510460251046025, 1195, 5))
    (50, (u'Usual Suspects, The (1995)', 4.505415162454874, 831, 5))
    (260, (u'Star Wars: Episode IV - A New Hope (1977)', 4.43953006219765, 1447, 4))
    (2762, (u'Sixth Sense, The (1999)', 4.4, 1110, 5))
    (1193, (u"One Flew Over the Cuckoo's Nest (1975)", 4.358816276202219, 811, 4))
    (593, (u'Silence of the Lambs, The (1991)', 4.358173076923077, 1248, 5))
    (2324, (u'Life Is Beautiful (La Vita \ufffd bella) (1997)', 4.325383304940375, 587, 5))
    (1136, (u'Monty Python and the Holy Grail (1974)', 4.324110671936759, 759, 5))
    (2571, (u'Matrix, The (1999)', 4.3096, 1250, 5))
    (1196, (u'Star Wars: Episode V - The Empire Strikes Back (1980)', 4.309457579972183, 1438, 4))
    (608, (u'Fargo (1996)', 4.286535303776683, 1218, 5))
    (1213, (u'GoodFellas (1990)', 4.282367447595561, 811, 5))
    (2858, (u'American Beauty (1999)', 4.27943661971831, 1775, 5))
    (2692, (u'Run Lola Run (Lola rennt) (1998)', 4.236263736263736, 546, 5))
    (3114, (u'Toy Story 2 (1999)', 4.232558139534884, 860, 5))
    (356, (u'Forrest Gump (1994)', 4.121270452358036, 1039, 5))
    (3147, (u'Green Mile, The (1999)', 4.107407407407408, 540, 5))
    
    Latent factors used to predict user ratings = 11
    Model RMSE on the test set = 0.89681898012
    
    Your top-rated 20 movies as predicted (for movies with more than 200 reviews):
    Movie ID, Movie Name, Predicted Rating, Average User Rating, Rating Count
    1. (1148, u'Wrong Trousers, The (1993)', 4.770542019260768, 4.531764705882353, 425)
    2. (745, u'Close Shave, A (1995)', 4.753631924974609, 4.490566037735849, 318)
    3. (2019, u'Seven Samurai (The Magnificent Seven) (Shichinin no samurai) (1954)', 4.703533430003581, 4.586330935251799, 278)
    4. (296, u'Pulp Fiction (1994)', 4.6286709639285135, 4.296438883541867, 1039)
    5. (2028, u'Saving Private Ryan (1998)', 4.619636891350235, 4.335826477187734, 1337)
    6. (2804, u'Christmas Story, A (1983)', 4.612181155215013, 4.2250755287009065, 662)
    7. (2329, u'American History X (1998)', 4.608706612983392, 4.219941348973607, 341)
    8. (1704, u'Good Will Hunting (1997)', 4.594344969591601, 4.182509505703422, 789)
    9. (3897, u'Almost Famous (2000)', 4.591743045486686, 4.228494623655914, 744)
    10. (750, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 4.591247076006126, 4.45468509984639, 651)
    11. (1250, u'Bridge on the River Kwai, The (1957)', 4.569505098361001, 4.390852390852391, 481)
    12. (913, u'Maltese Falcon, The (1941)', 4.568127212954204, 4.388655462184874, 476)
    13. (1263, u'Deer Hunter, The (1978)', 4.547783894275325, 4.063091482649842, 317)
    14. (908, u'North by Northwest (1959)', 4.5463688346057545, 4.394285714285714, 700)
    15. (1207, u'To Kill a Mockingbird (1962)', 4.544229159599917, 4.450110864745011, 451)
    16. (3435, u'Double Indemnity (1944)', 4.530356551120931, 4.423357664233577, 274)
    17. (1221, u'Godfather: Part II, The (1974)', 4.519823977024758, 4.363975155279503, 805)
    18. (1299, u'Killing Fields, The (1984)', 4.516667729666294, 4.289017341040463, 346)
    19. (1242, u'Glory (1989)', 4.512914869520004, 4.216757741347905, 549)
    20. (1223, u'Grand Day Out, A (1992)', 4.506497073894002, 4.351351351351352, 222)
    
    
    ************* Hypothesis testing on Recommender System completed *************

