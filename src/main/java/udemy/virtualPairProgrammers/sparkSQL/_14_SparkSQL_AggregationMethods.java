package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import static org.apache.spark.sql.functions.*;

public class _14_SparkSQL_AggregationMethods {

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

        Dataset<Row> maxValue = inputData.groupBy("subject").max("score").alias("max_score");
        maxValue.show(false);

        // [The main method is the agg function, which has multiple variants.](https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/RelationalGroupedDataset.html)
        Dataset<Row> aggregatedValues = inputData.groupBy("subject").agg(
                functions.max("score").cast(DataTypes.IntegerType).alias("max_score"),
                functions.min("score").cast(DataTypes.IntegerType).alias("min_score"),
                functions.avg("score").cast(DataTypes.DoubleType).alias("avg_score")
        );
        aggregatedValues.show(false);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }

}
