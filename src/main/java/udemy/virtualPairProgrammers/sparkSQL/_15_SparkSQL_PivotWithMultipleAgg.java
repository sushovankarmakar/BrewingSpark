package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.Scanner;

import static org.apache.spark.sql.functions.*;

public class _15_SparkSQL_PivotWithMultipleAgg {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-sql")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        Dataset<Row> inputData = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true) // WARNING: inferSchema causes one full scan of the data(https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/DataFrameReader.html)
                .csv("src/main/resources/udemy/virtualPairProgrammers/exams/students.csv");

        // [The main method is the agg function, which has multiple variants.](https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/RelationalGroupedDataset.html)

        // Build a pivot table, showing each subject down the 'left-hand side' and years 'across the top'
        // for each subject and year, we want
        // the average exam score
        // the standard deviation of scores (stddev)
        // all values up to 2 decimal places

        Dataset<Row> aggregatedValues = inputData.groupBy("subject")
                .pivot("year")
                .agg(
                        functions.round(avg(col("score")), 2).alias("avg_score"),
                        functions.round(stddev(col("score")), 2).alias("std_dev")
                );
        aggregatedValues.show(false);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }
}
