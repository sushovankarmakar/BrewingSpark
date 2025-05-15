package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class _11_SparkSQL_InMemoryData {

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

        // ------------------------- Converting JavaRDD to Dataset -------------------------
        convertingJavaRDDToDataset(sparkSession, inMemoryData, schema);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }

    private static void convertingJavaRDDToDataset(SparkSession sparkSession, List<Row> inMemoryData, StructType schema) {
        System.out.println("Converting JavaRDD to Dataset");
        try (JavaSparkContext jsc = new JavaSparkContext(sparkSession.sparkContext())) {

            // Step 2: Create a JavaRDD from the inMemoryData
            JavaRDD<Row> javaRDD = jsc.parallelize(inMemoryData);

            // Step 3: Convert the JavaRDD back into a Dataset
            Dataset<Row> dataset = sparkSession.createDataFrame(javaRDD, schema);
            dataset.show(5);
        }
    }
}
