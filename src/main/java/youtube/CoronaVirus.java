package youtube;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import java.util.*;

public class CoronaVirus {

    // https://www.youtube.com/watch?v=_kFNxF2MM_M&list=PL3N9eeOlCrP5PfpYrP6YxMNtt5Hw27ZlO&index=8&ab_channel=AIEngineering
    // I followed the above YouTube video to implement below code.
    // sample data taken from : https://github.com/srivatsan88/YouTubeLI/tree/master/dataset/coronavirus

    // I've learnt the below topics
    // 1. how to load the data.
    // 2. diff between RDD vs Dataset
    // 3. map vs flatmap
    // 4. select, filter
    // 5. explain
    // 6. sort, orderBy, sortWithinPartitions
    // 7. describe
    // 8. printSchema
    // 9. approxQuantile
    // 10. agg
    // 11. join
    // 12. Window.partitionBy(), WindowSpec
    // 13. withColumn
    // 14. pivot, crosstab
    // 15. minus function

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("corona-virus")
                .setMaster("local[*]");

        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        //JavaRDD<String> dataset = jsc.textFile("src/main/resources/corona_dataset_latest.csv");

        SparkSession spark = SparkSession.builder().config(sparkConf).getOrCreate();

        String coronaInputPath = "src/main/resources/youtube/corona_dataset.csv";
        Dataset<Row> coronaDataset = spark.read()
                .option("inferSchema", true)
                .option("header", true)
                .csv(coronaInputPath);

        //coronaDataset.show(10, false);

        String twitterInputPath = "src/main/resources/youtube/tweets.csv";
        Dataset<Row> twitterDataset = spark.read()
                .option("inferSchema", true)
                .option("header", true)
                .csv(twitterInputPath);

        Dataset<Row> filteredData = twitterDataset.filter("country='USA'");// AND location like 'New%'
        filteredData.explain();

        Dataset<Row> locationStartsWithN = twitterDataset.select(functions.col("location").startsWith("N"));
        locationStartsWithN.explain();

        twitterDataset.toJavaRDD().map(new Function<Row, Object>() {
            @Override
            public Object call(Row row) throws Exception {
                return null;
            }
        });

        // For better performance, prefer using Dataset's .map() directly
        // instead of converting to RDD first, as it benefits from Spark SQL's optimizations.
        Dataset<String> mapped = twitterDataset.map(new MapFunction<Row, String>() {
            @Override
            public String call(Row row) throws Exception {
                return row.getString(2).split(" ")[0];
            }
        }, Encoders.STRING()); // you need to provide an Encoder to tell Spark how to serialize/deserialize the output type

        System.out.println("Collect as list : " + mapped.takeAsList(10));

        Dataset<String> words = twitterDataset.flatMap(new FlatMapFunction<Row, String>() {
            @Override
            public Iterator<String> call(Row row) throws Exception {
                return Arrays.stream(row.getString(2).split(" ")).iterator();
            }
        }, Encoders.STRING());

        System.out.println("Words : " + words.takeAsList(10));


        //twitterDataset.show(10, false);

        // rdd vs dataset - rdd is old, dataset is new - use tungsten execution engine
        // to convert rdd to dataset -> we need to pass a schema (StructType) with it. spark.createDataFrame(rowRDD, schema);

        /// Datasets and DataFrames
        //A Dataset is a distributed collection of data.
        // Dataset is a new interface added in Spark 1.6
        // that provides the benefits of RDDs (strong typing, ability to use powerful lambda functions)
        // with the benefits of Spark SQL’s optimized execution engine
        // A DataFrame is a Dataset organized into named columns.
        // It is conceptually equivalent to a table in a relational database or a data frame in R/Python,
        // but with richer optimizations under the hood.

        // narrow transformation : map, filter - no shuffle
        // wide transformation : reduceByKey - shuffle across partitions
        // map vs flatmap : map = 1:1, flatmap = 1:n

        JavaRDD<Integer> listRDD = jsc.parallelize(Arrays.asList(1, 4, 8));

        JavaRDD<Integer> mapRDD = listRDD.map(val -> val * val);

        System.out.println(mapRDD.collect());

        java.util.List<Integer> collect2 = listRDD.flatMap(new FlatMapFunction<Integer, Integer>() {
            @Override
            public Iterator<Integer> call(Integer integer) throws Exception {

                ArrayList<Integer> vals = new ArrayList<>();
                vals.add(integer * integer);
                return vals.iterator();
            }
        }).collect();
        System.out.println(collect2);


        // Multiple columns with different orders
        coronaDataset.filter("Country='US'")
                .sort(functions.col("Date").desc(), functions.col("state_cleaned").asc())
                .show(5);

        // You can also use orderBy() instead of sort() - they are equivalent
        coronaDataset.filter("Country='US'")
                .orderBy(functions.col("Date").desc(), functions.col("state_cleaned").asc())
                .show(5);

        // sort and orderBy are very expensive operations - wide transformation and a lot of shuffles
        // to reduce the shuffling, use sortWithinPartitions
        coronaDataset.filter("Country='US'")
                .sortWithinPartitions(functions.col("Date").desc(), functions.col("state_cleaned").asc())
                .show(5);


        coronaDataset.describe().show(); // showing stats for all the columns in the dataset

        coronaDataset.printSchema(); // printing schema - we were reading the data, we gave (inferSchema, true) in option

        coronaDataset.filter("Confirmed > 10000")
                .sort(functions.col("Confirmed").desc())
                .show(10);

        // Calculate quantiles for "Confirmed" column
        // Parameters: column name, probabilities array, relative error
        double[] quantiles = coronaDataset.stat().approxQuantile(
                "Confirmed", // column name
                new double[]{0.25, 0.5, 0.75, 0.9, 0.95}, // probabilities for quartiles
                0.9 // relative error tolerance
        );

        System.out.println("25th percentile: " + quantiles[0]);
        System.out.println("Median: " + quantiles[1]);
        System.out.println("75th percentile: " + quantiles[2]);
        System.out.println();

        Map<String, String> map = new HashMap<>();
        map.put("Date", "max");
        /*map.put("confirmed", "max");
        map.put("Death", "min");
        map.put("Recovered", "max");*/
        java.util.List<Row> rows = coronaDataset.agg(map).collectAsList();

        Date maxDate = rows.get(0).getDate(0);
        System.out.println("maxDate : " + maxDate);

        System.out.println("Max date for each country");
        Dataset<Row> maxDateByCountry = coronaDataset.groupBy(functions.col("Country"), functions.col("state_cleaned"))
                .agg(functions.max("Date").alias("Date"));

        // also possible : .withColumnRenamed("max(Date)", "Date");

        maxDateByCountry.show();

        Dataset<Row> confirmedMaxDataset = coronaDataset.join(maxDateByCountry, new String[]{"Country", "state_cleaned", "Date"})
                .sort(functions.col("Confirmed").desc());
        confirmedMaxDataset.show();

        // Define a window specification
        WindowSpec windowSpec = Window.partitionBy(functions.col("Country"), functions.col("state_cleaned"))
                .orderBy(functions.col("Date").desc());

        // Add row number and running totals
        Dataset<Row> windowedData = coronaDataset.withColumn("row_number", functions.row_number().over(windowSpec));

        windowedData.show();

        // Show only first row per country to get latest data
        windowedData.filter("row_number = 1")
                .show();

        // columnar view of time series data
        coronaDataset.groupBy(functions.col("Country"))
                .pivot(functions.col("Date"))
                .agg(functions.sum("Confirmed"))
                .show();

        coronaDataset.filter("Country == 'US'")
                .stat()
                .crosstab("State", "Date")
                .show();

        confirmedMaxDataset.groupBy("Country")
                .agg(functions.sum("Confirmed"), functions.sum("Death"), functions.sum("Recovered"))
                .orderBy(functions.sum("Confirmed").desc())
                .show();

        // adding a new column named 'Active'
        confirmedMaxDataset = confirmedMaxDataset.withColumn("Active",
                (confirmedMaxDataset.col("Confirmed")
                        .minus(confirmedMaxDataset.col("Recovered"))
                        .minus(confirmedMaxDataset.col("Death")))
        );
        confirmedMaxDataset.show();

        confirmedMaxDataset.groupBy(functions.col("Country"))
                .sum("Active")
                .orderBy(functions.col("sum(Active)").desc())
                .show();

        jsc.close();
    }
}
