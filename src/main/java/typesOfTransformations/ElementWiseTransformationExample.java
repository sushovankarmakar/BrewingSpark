package typesOfTransformations;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.List;

public class ElementWiseTransformationExample {
    public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local", "typesOfTransformations.ElementWiseTransformationExample");

        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
        JavaRDD<Integer> rdd = sc.parallelize(data);

        // Example of map transformation
        JavaRDD<Integer> squaredRDD = rdd.map(x -> x * x);
        System.out.println("Squared RDD: " + squaredRDD.collect());

        // Example of filter transformation
        JavaRDD<Integer> evenRDD = rdd.filter(x -> x % 2 == 0);
        System.out.println("Even RDD: " + evenRDD.collect());

        sc.close();
    }
}