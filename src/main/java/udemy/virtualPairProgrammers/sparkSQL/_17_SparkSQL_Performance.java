package udemy.virtualPairProgrammers.sparkSQL;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import java.util.Scanner;

@Slf4j
public class _17_SparkSQL_Performance {

    // Datatypes decides HashAggregation or SortAggregation
    // 1. HashAggregation is faster than SortAggregation.
    //    1. Because, in HashAggregation we do NOT have to do sorting before doing aggregation
    //
    // 2. HashAggregation internally used map
    //    1. In this map,
    //        1. Key is built from those columns which are part of ‘group by’ columns and
    //        2. Values are built from columns which are NOT part of ‘group by’ columns
    //    2. Which is directly created in memory, and this map is not a part of Java Object.
    //    3. This map internally uses UnsafeRow functionality and
    //    4. UnsafeRow allows only some kind of datatypes, and a String type is not part of that.
    //
    // 3. So, based on data types of the columns which are a part of group by, HashAggregation or SortAggregation is triggered.

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-performance")
                .setMaster("local[*]");

        // spark.sql

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .config("spark.sql.shuffle.partitions", "4")
                .getOrCreate();

        Dataset<Row> dataset = sparkSession.read()
                .option("header", true)
                //.csv("src/main/resources/udemy/virtualPairProgrammers/bigfiles/bigLog.txt");
                .csv("src/main/resources/udemy/virtualPairProgrammers/biglog.txt");

        dataset.createOrReplaceTempView("logging_table");

        // SPARK SQL: -----------------------------------------------------------------------
        // 1.
        // for grouping, here the below SQL is using the sort aggregation because
        // this spark SQL will use SortAggregation because here all the columns, which are not part of group by columns,
        // those are NOT immutable types, as you can see,
        // cast(first(date_format(datetime, 'M')) AS int) -> this produces a string type.

        /*dataset = sparkSession.sql("SELECT level, " +
                "date_format(datetime, 'MMM') AS month, " +
                "count(1) AS total " +
                "FROM logging_table " +
                "GROUP BY level, month " +
                "ORDER BY cast(first(date_format(datetime, 'M')) AS int), level");*/

        // 2.
        // this spark SQL will use HashAggregation because here all the columns, which are not part of group by columns,
        // those are immutable types
        // we've changed the monthNum as integer from string.
        // as integer is supported by UnsafeRow and UnsafeRow is used by HashAggregation.
        // so, here HashAggregation is used over SortAggregation

        dataset = sparkSession.sql("SELECT level, " +
                "date_format(datetime, 'MMM') AS month, " +
                "count(1) AS total, " +
                "first(cast(date_format(datetime, 'M') AS int)) AS monthNum " +
                "FROM logging_table " +
                "GROUP BY level, month " +
                "ORDER BY monthNum, level");

        dataset = dataset.drop("monthNum");

        // JAVA API: -----------------------------------------------------------------------

        /*dataset = dataset.select(functions.col("level"),
                functions.date_format(functions.col("datetime"), "MMM").alias("month"),
                functions.date_format(functions.col("datetime"), "M").alias("monthNum").cast(DataTypes.IntegerType)
        );

        dataset = dataset.groupBy("level", "month", "monthNum")
                .count()
                .as("total")
                .orderBy("monthNum");

        dataset = dataset.drop("monthNum");*/

        long startTime = System.currentTimeMillis();

        dataset.show(100);
        dataset.explain();

        System.out.println("Time taken " + (System.currentTimeMillis() - startTime) + " ms");

        /*Scanner scanner = new Scanner(System.in);
        scanner.nextLine();*/

        sparkSession.close();
    }
}
