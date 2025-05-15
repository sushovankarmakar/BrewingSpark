package typesOfTransformations;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class PartitionWiseTransformationExample {
    public static void main(String[] args) {
        JavaSparkContext sc = new JavaSparkContext("local", "typesOfTransformations.PartitionWiseTransformationExample");

        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
        JavaRDD<Integer> rdd = sc.parallelize(data, 2);

        // Example of mapPartitions transformation
        JavaRDD<Integer> partitionSumRDD = rdd.mapPartitions((Iterator<Integer> iter) -> {
            int sum = 0;
            while (iter.hasNext()) {
                sum += iter.next();
            }
            return Arrays.asList(sum).iterator();
        });
        System.out.println("Partition Sum RDD: " + partitionSumRDD.collect());

        sc.close();
    }
}