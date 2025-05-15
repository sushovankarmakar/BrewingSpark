package practice;

import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.List;

public class MapPractice {

    public static void main(String[] args) {

        List<String> inputData = new ArrayList<>();
        inputData.add("abc");
        inputData.add("sushovan");
        inputData.add("world-cup");

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.setAppName("practice");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        SparkContext sc = sparkSession.sparkContext();

        try (JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc)) {
            JavaRDD<String> javaRDD = jsc.parallelize(inputData);

            // Add a Suffix to Each String
            // Convert Strings to Uppercase
            // Remove Leading and Trailing Spaces from String -> trim()
            // Calculate Length of Each String
            // Extract First Character of Each String
            // Reverse Each String
            JavaRDD<String> upperCaseRDD = javaRDD.map(val ->
                    "God's plan baby, "
                            + val.toUpperCase().trim()
                            + ", " + val.length()
                            + ", " + val.charAt(0)
                            + ", " + StringUtils.reverse(val)
            );

            Dataset<String> dataset = sparkSession.createDataset(upperCaseRDD.rdd(), Encoders.STRING());
            dataset.show(false);
        }
    }
}
