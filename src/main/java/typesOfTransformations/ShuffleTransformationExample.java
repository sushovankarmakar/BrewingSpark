package typesOfTransformations;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

public class ShuffleTransformationExample {
    public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local", "typesOfTransformations.ShuffleTransformationExample");

        List<Tuple2<Integer, String>> data1 = Arrays.asList(
                new Tuple2<>(1, "A"),
                new Tuple2<>(2, "B"),
                new Tuple2<>(1, "C"),
                new Tuple2<>(3, "D")
        );
        List<Tuple2<Integer, Integer>> data2 = Arrays.asList(
                new Tuple2<>(1, 10),
                new Tuple2<>(2, 20),
                new Tuple2<>(1, 30),
                new Tuple2<>(3, 40)
        );

        JavaPairRDD<Integer, String> rdd1 = sc.parallelizePairs(data1);
        JavaPairRDD<Integer, Integer> rdd2 = sc.parallelizePairs(data2);

        // Example of groupByKey transformation
        JavaPairRDD<Integer, Iterable<String>> groupedRDD = rdd1.groupByKey();
        System.out.println("Grouped RDD: " + groupedRDD.collect());

        // Example of reduceByKey transformation
        JavaPairRDD<Integer, Integer> reducedRDD = rdd2.reduceByKey((x, y) -> x + y);
        System.out.println("Reduced RDD: " + reducedRDD.collect());

        // Example of join transformation
        JavaPairRDD<Integer, Tuple2<String, Integer>> joinedRDD = rdd1.join(rdd2);
        System.out.println("Joined RDD: " + joinedRDD.collect());

        sc.close();
    }
}