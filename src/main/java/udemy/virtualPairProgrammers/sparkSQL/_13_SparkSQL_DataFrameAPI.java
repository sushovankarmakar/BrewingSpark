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

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.date_format;

public class _13_SparkSQL_DataFrameAPI {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-sql")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        List<Row> inMemoryData = new ArrayList<>();
        inMemoryData.add(RowFactory.create("WARN", "2016-12-31 04:19:32"));
        inMemoryData.add(RowFactory.create("FATAL", "2016-12-31 03:22:34"));
        inMemoryData.add(RowFactory.create("WARN", "2016-12-31 03:21:21"));
        inMemoryData.add(RowFactory.create("INFO", "2015-4-21 14:32:21"));
        inMemoryData.add(RowFactory.create("FATAL", "2015-4-21 19:23:20"));

        StructType schema = new StructType(new StructField[]{
                new StructField("level", DataTypes.StringType, false, Metadata.empty()),
                new StructField("datetime", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> inputData = sparkSession.createDataFrame(inMemoryData, schema);
        inputData.show(5);

        /*Dataset<Row> dataByMonths = sparkSession.sql("SELECT" +
                        " level" +
                        ", date_format(datetime, 'MMMM') AS month" +
                        ", CAST(FIRST(date_format(datetime, 'M')) AS int) AS monthNum" +
                        ", COUNT(*) AS totalPerMonth" +
                        " FROM logging_table" +
                        " GROUP BY level, month" +
                        " ORDER BY monthNum ASC, level DESC")
                .drop("monthNum");*/

        Dataset<Row> dataByMonths = inputData.select(
                        col("level"),
                        date_format(col("datetime"), "MMMM").alias("month"),
                        date_format(col("datetime"), "M").cast(DataTypes.IntegerType).alias("monthNum"))
                .groupBy(col("level"), col("month"), col("monthNum")) // aggregation returns RelationalGroupedDataset
                .count()
                .orderBy(col("monthNum").asc(), col("level").desc())
                .drop("monthNum");

        dataByMonths.show(false);

        // ----------------------------------- pivot table example -----------------------------------
        pivotTableExample(inputData);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }

    private static void pivotTableExample(Dataset<Row> inputData) {

        // https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/RelationalGroupedDataset.html#pivot-org.apache.spark.sql.Column-java.util.List-

        String[] monthNames = new String[]{"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
        List<Object> months = Arrays.asList(monthNames);

        Dataset<Row> pivotData = inputData.select(
                        col("level"),
                        date_format(col("datetime"), "MMMM").alias("month"),
                        date_format(col("datetime"), "M").cast(DataTypes.IntegerType).alias("monthNum"))
                .groupBy(col("level"))
                .pivot(col("monthNum")) // we can pass a list of column names as second parameter, which will be shown in the output
                .count()
                .na().fill(0); // replace null values with 0

        pivotData.show(false);
    }

}
