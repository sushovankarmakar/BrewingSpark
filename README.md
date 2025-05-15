[RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)

## 12.04.2025

### In-memory data
* In-memory input data is _helpful in unit testing_ when we need to test our code against a small amount of input data
```java
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

List<Row> inMemoryData = new ArrayList<>();
inMemoryData.add(RowFactory.create("WARN", "2016-12-31 04:19:32"));
inMemoryData.add(RowFactory.create("FATAL", "2016-12-31 03:22:34"));

StructType schema = new StructType(new StructField[]{
        new StructField("level", DataTypes.StringType, false, Metadata.empty()),
        new StructField("datetime", DataTypes.StringType, false, Metadata.empty())
});

Dataset<Row> inputData = sparkSession.createDataFrame(inMemoryData, schema);
inputData.show(5);
```

### Converting JavaRDD to Dataset
```java
try (JavaSparkContext jsc = new JavaSparkContext(sparkSession.sparkContext())) {

    // Step 2: Create a JavaRDD from the inMemoryData
    JavaRDD<Row> javaRDD = jsc.parallelize(inMemoryData);
    
    // Step 3: Convert the JavaRDD back into a Dataset
    Dataset<Row> dataset = sparkSession.createDataFrame(javaRDD, schema);
    dataset.show(5);
}
```


## 11.04.2025

## SparkSQL in Java

### Reading CSV files using SparkSQL
The `SparkSession` API provides an easy way to read CSV files and work with them as `Dataset<Row>` objects.

#### **Key Points**:
1. SparkSession: The entry point for working with structured data in Spark.
2. Dataset<row></row>: Represents tabular data similar to a table in a database.
3. Options:
   * header: Set to true to use the first row as column names.
   * inferSchema: Automatically infers the data types of columns (optional).

#### **Benefits of SparkSQL**:
1. Simplifies working with structured data.
2. Provides SQL-like operations for querying data.
3. Integrates seamlessly with Spark's RDD and DataFrame APIs.

#### **Additional Notes:**
* Use `.config("spark.sql.warehouse.dir", "file:///c:/tmp/")` to specify a warehouse directory for SparkSQL if needed.
* SparkSQL is ideal for working with large datasets and performing complex queries.

#### Example:
```java
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkSQLExample {

    public static void main(String[] args) {

        String inputFilePath = "src/main/resources/udemy/virtualPairProgrammers/exams/students.csv";

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-sql")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        // Reading CSV file into a Dataset
        Dataset<Row> inputData = sparkSession.read()
                .option("header", "true") // Use the first row as a header
                .csv(inputFilePath);

        // Displaying the data
        inputData.show();

        // Counting the total number of rows
        System.out.println("Total count: " + inputData.count());
    }
}
```

#### Filter using sql expression

```java
System.out.println("Filter using sql expression");
Dataset<Row> modernArtResults = inputData.filter("subject = 'Modern Art' AND year >= 2007 ");
modernArtResults.show(5);
```
---

####  Filter using lambda function

```java
System.out.println("Filter using lambda function");
Dataset<Row> moderArtResults1 = inputData.filter(new FilterFunction<Row>() {
    @Override
    public boolean call(Row row) throws Exception {
        return row.getAs("subject").equals("Modern Art")
                && Integer.parseInt(row.getAs("year")) >= 2007;
    }
});
moderArtResults1.show(5);
```
---

#### Filter using column class
Version 1 : 
```java
System.out.println("Filter using column class : version 1");
Column subject = inputData.col("subject");
Column year = inputData.col("year");

Dataset<Row> modernArtResults2 = inputData.filter(subject.equalTo("Modern Art")
        .and(year.geq(2007))
);
modernArtResults2.show(5);
```

Version 2 : using functions.col : domain specific language style
```java
import static org.apache.spark.sql.functions.col;

System.out.println("Filter using column class : version 2");
Column subject1 = col("subject");
Column year1 = col("year");

Dataset<Row> modernArtResults3 = inputData.filter(subject1.equalTo("Modern Art")
        .and(year1.geq(2007))
);
modernArtResults3.show(5);
```
---

#### Spark Temporary View
View is an in-memory table structure created from a dataset where we can perform full SQL operations
```java
inputData.createOrReplaceTempView("students");
Dataset<Row> modernArtResults = sparkSession.sql("SELECT * FROM students s WHERE s.subject = 'Modern Art' AND s.year >= 2007");
modernArtResults.show(5);

// here we're calling distinct, shuffle will happen.
Dataset<Row> distinctYears = sparkSession.sql("SELECT distinct year FROM students s");
distinctYears.show();
```

----------------------------------------
## 08.04.2025

### Wide and Narrow transformation.

#### Narrow transformation
1. filter, mapToPair - these are narrow transformation.

#### Wide transformation 
1. Wide transformation produces shuffling so try to delay wide transformation as late as possible in the DAG.
2. groupByKey, join - these are wide transformation

### avoid groupByKey and use 'map-side-reduce' instead
* reduceByKey is better than groupByKey although both of them will cause shuffling
* but still reduceByKey is better cause
* reduceByKey has two stages
  * at first stage : it will apply the reduce function on each partition which does NOT reduce any shuffling.
  * at second stage : it will shuffle to bring similar keys into one partition but this shuffling is very minimal.
* reduceByKey is sometimes called as 'map-side-reduce'.
----------------------------------------
1. [How To Use Spark Transformations Efficiently For MapReduce-Like Jobs](https://www.finra.org/about/technology/blog/how-to-use-spark-transformations-efficiently-for-mapreduce-like-jobs)
2. [Spark Reduction Transformations Explained](https://scaibu.medium.com/spark-reduction-transformations-explained-b7642c668d01)
3. [Difference between reduceByKey and groupByKey](https://www.linkedin.com/pulse/apache-spark-difference-between-reducebykey-abhijit-sarkhel/)

---
## 02.04.2025

* Reading txt file vs csv file using `javaSparkContext.textFile("filePath")` : [Explanation](src/main/java/udemy/virtualPairProgrammers/_8_BigDataExercise.md)

----------------------------------------

## 31.03.2025

### Joins

#### 1.  Inner join
```java
JavaPairRDD<Integer, Tuple2<Integer, String>> innerJoin = userVisits.join(users);
```
```
Inner join
(6, (4,  Raquel)) 
(4, (18, Doris)) 
```
----

#### 2. Left outer join
```java
JavaPairRDD<Integer, Tuple2<Integer, Optional<String>>> leftOuterJoin = userVisits.leftOuterJoin(users);
```
```
(10, (9,    Optional.empty)) 
(6,  (4,    Optional[Raquel])) 
(4,  (18,   Optional[Doris])) 
```
----
#### 3. Right outer join
```java
JavaPairRDD<Integer, Tuple2<Optional<Integer>, String>> rightOuterJoin = userVisits.rightOuterJoin(users);
```
```
(5, (Optional.empty,    Marybelle)) 
(3, (Optional.empty,    Alan)) 
(1, (Optional.empty,    John)) 
(2, (Optional.empty,    Bob)) 
(6, (Optional[4],       Raquel)) 
(4, (Optional[18],      Doris))
```
----

#### 4. Full outer join
```java
JavaPairRDD<Integer, Tuple2<Optional<Integer>, Optional<String>>> fullOuterJoin = userVisits.fullOuterJoin(users);
```
```
(5,     (Optional.empty,    Optional[Marybelle])) 
(3,     (Optional.empty,    Optional[Alan])) 
(10,    (Optional[9],       Optional.empty)) 
(4,     (Optional[18],      Optional[Doris])) 
(2,     (Optional.empty,    Optional[Bob])) 
(1,     (Optional.empty,    Optional[John])) 
(6,     (Optional[4],       Optional[Raquel])) 
```

----

#### 5. Cross Join or Cartesian
```java
JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Integer, String>> cartesian = userVisits.cartesian(users);
```
* All the values will be paired up with every single possible combination
```
((4,  18),   (1, John)) 
((4,  18),   (3, Alan)) 
((4,  18),   (2, Bob)) 
((4,  18),   (4, Doris)) 
((4,  18),   (5, Marybelle)) 
((4,  18),   (6, Raquel)) 
((6,  4),    (2, Bob)) 
((6,  4),    (1, John)) 
((6,  4),    (4, Doris)) 
((6,  4),    (6, Raquel)) 
((6,  4),    (3, Alan)) 
((6,  4),    (5, Marybelle)) 
((10, 9),    (1, John)) 
((10, 9),    (2, Bob)) 
((10, 9),    (3, Alan)) 
((10, 9),    (4, Doris)) 
((10, 9),    (6, Raquel)) 
((10, 9),    (5, Marybelle))
```
----

### 30.03.2025

#### Practical : Keywords ranking
1. Load the file into RDD 
    ```java
    jsc.textFile(inputFile)
    ```
2. Filtering all noisy lines i.e. timestamps, empty lines and turning all of them in lowercase
    ```java
    .map(line -> line.replaceAll("[^a-zA-Z\\s]", "").trim().toLowerCase())
    ```
3. Converting lines into words
   ```java
    .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
    ```
4. Filtering out the empty AND boring words
   ```java
    .filter(word -> !word.isEmpty() && Util.isNotBoring(word))
    ```
5. Count up the remaining words
   ```java
    .mapToPair(word -> new Tuple2<>(word, 1L))
    .reduceByKey((v1, v2) -> v1 + v2)
    ```
6. Find the top 10 most frequently used words
    ```java
    .mapToPair(pair -> new Tuple2<>(pair._2, pair._1)) // switching value and key so that we can sort by value
    .sortByKey(false)
    .take(10)
    ```
7. Final output
    ```java
    JavaPairRDD<Long, String> wordFrequencies = jsc.textFile(inputFileSpringCourseSubtitles)
        .map(line -> line.replaceAll("[^a-zA-Z\\s]", "").trim().toLowerCase())
        .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
        .filter(word -> !word.isEmpty() && Util.isNotBoring(word))
        .mapToPair(word -> new Tuple2<>(word, 1L)) // take PairFunction
        .reduceByKey((v1, v2) -> v1 + v2) // takes Function2
        .mapToPair(pair -> new Tuple2<>(pair._2, pair._1))  // switching value and key so that we can sort by value
        .sortByKey(false); // sorting in descending order

    wordFrequencies
        .take(10)
        .forEach(System.out::println);
    ```

### 29.03.2025

#### Reading input files from disk
* In big data, we usually read from a distributed file system like s3 or hadoop
* when we call textFile, it doesn't store the entire file into the memory
* Instead, textFile() asked driver to inform its workers to load parts (partitions) of this input file
```java
JavaRDD<String> inputRDD = jsc.textFile("...input-docker-course.txt");
```

### 28.03.2025

#### FlatMaps :
* Map function works in `(1 input --> 1 output)` format.
* But sometimes we need function to work with `(1 input --> 0 or more outputs)` format. Here comes flatmap
```java
JavaRDD<String> words = inputRdd.flatMap(val ->
      Arrays.asList(val.split(" ")).iterator() // flatMap returns an Iterator objects
);
```

### Filter :
* filtering out
```java
// filtering out numeric words
JavaRDD<String> filteredWords = words.filter(word -> !word.matches("\\d+"));
```

--------------------------------------------------------------------
### 27.03.2025

#### Tuple : [Link](./src/main/java/udemy/_3_Tuples.java)

* Keep different kinds of data in a same RDD.
* Can contain any kinds of data, including Java objects
* It is a scala concept.
* Tuple22 is the limit

```java
import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

JavaRDD<Tuple2<Integer, Double>> tupleRdd = myRDD.map(value -> new Tuple<>(value, Math.sqrt(value)));
```
--------------------------------------------------------------------
`Tuple` can store from 2 values to 22 values. <br>
To store 2 value, we need `Tuple2<Val1, Val2>`

Now storing 2 values (one key, another one is value) is so common that, <br>
We've a special Object called `PairRDD<Key, Value>`

And this `PairRDD` has some special methods to work with keys like <br>
- `aggregateByKey`
- `reduceByKey`
- `groupByKey`
--------------------------------------------------------------------

#### PairRDD : [Link](./src/main/java/udemy/_4_PairRDD.java)

* Call `mapToPair` method to get this
* It is common to store two values in a RDD, so we've `PairRDD`
* similar to Java `Map<Key, Value>` but here **key can be repeated**
* **Has extra methods like `groupByKey` and `reduceByKey` which operates specifically on key**
```java
import org.apache.spark.api.java.JavaPairRDD;
import scala.Tuple2;

JavaPairRDD<String, String> pairRDD = myRDD.mapToPair(val -> new Tuple2<>(key, value));
```
---
`groupByKey` : 
  * can lead to severe performance problems !!
  * can cause to skewness of the data

```java
import org.apache.spark.api.java.JavaPairRDD;

JavaPairRDD<String, Iterable<String>> groupByKey = pairRDD.groupByKey();
```
---

`reduceByKey` :
  * internally it does `map` and then `reduce`
    * the keys are gather together individually, before the reduction method is applied on its values.
    * values of key1 is gathered -> now apply to reduce function
    * values of key2 is gathered -> now apply to reduce function
    * and so on ...
  * performance wise, better than `groupByKey`

```java
import org.apache.spark.api.java.JavaPairRDD;

JavaPairRDD<Integer, Integer> reduceByKey = pairRDD.reduceByKey((v1, v2) -> v1 + v2);
```

#### Fluent API : [Link](./src/main/java/udemy/_4_PairRDD_WithFluentAPI.java)
* one function's output can be another function's input. so we can do chaining
```java
JavaSparkContext sc = new JavaSparkContext(sparkConf);

sc.parallelize(inputData)                                          // returns JavaRDD<String>
  .mapToPair(rawValue -> new Tuple2<>(rawValue.split(":")[0], 1L)) // returns JavaPairRDD<String, Integer>
  .reduceByKey((val1, val2) -> val1 + val2)                          // returns JavaPairRDD<String, Integer>
  .foreach(tuple -> System.out.println(tuple._1 + " has " + tuple._2 + " records")); // returns void
```
--------------------------------------------------------------------

### 25.03.2025
#### Map : [Link](./src/main/java/udemy/_2_Function_Map.java)

```java
import org.apache.spark.api.java.JavaRDD;

JavaRDD<Double> squareRoots = myRDD.map(value -> Math.sqrt(value));
```  
#### Reduce : [Link](./src/main/java/udemy/_1_Function_Reduce.java)
* to transform a big dataset into a single answer  
```java
Double sum = myRDD.reduce((val1, val2) -> va1 + val2);
```

  Rdd is always immutable. Once created, it can't be changed.
