package udemy.virtualPairProgrammers.sparkSQL;

import com.google.common.collect.Iterables;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.Scanner;

public class _9_PartitionTesting {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setMaster("local[*]")
                .setAppName("partition-testing");

        try (JavaSparkContext jsc = new JavaSparkContext(conf)) {
            JavaRDD<String> javaRDD = jsc.textFile("src/main/resources/udemy/virtualPairProgrammers/biglog.txt"); // not going to commit this big 300 MB file.

            System.out.println("InitialRDD has : " + javaRDD.getNumPartitions() + " number of partitions.");

            // 'Salting' just mangles the key to artificially spread them across partitions
            // It means, you'll have at some point do the work to group them back together.
            // It is the last resort, but it works
            JavaPairRDD<String, String> pairRDD = javaRDD.mapToPair(inputLine -> {
                String[] parts = inputLine.split(",");
                String level = parts[0]; //  + (int)(Math.random() * 11); here we can do 'salting' on the key so that keys are evenly distributed.
                String date = parts[1];
                return new Tuple2<>(level, date);
            });
            System.out.println("After a narrow transformation, we've : " + pairRDD.getNumPartitions() + " number of partitions.");

            // Now we're going to do a "wide" transformation
            JavaPairRDD<String, Iterable<String>> results = pairRDD.groupByKey();
            System.out.println("After a wide transformation, we've : " + results.getNumPartitions() + " number of partitions.");

            //results = results.cache(); // cache() would only work if there is enough memory to hold the data. so use it if our data size is very small.

            results = results.persist(StorageLevel.MEMORY_AND_DISK());

            results.foreach(it -> System.out.println("key " + it._1 + " has " + Iterables.size(it._2) + " elements."));

            System.out.println("Total count is : " + results.count());

            // Spark runs a UI server on port 4040, and this server is ONLY accessible through the lifespan of our code.
            // This below scanner code is used to stop completing our code so that we can look into the Spark UI peacefully.
            Scanner sc = new Scanner(System.in);
            sc.nextLine();
            sc.close();


            // avoid groupByKey and use 'map-side-reduce' instead

            // reduceByKey is better than groupByKey although both of them will cause shuffling
            // but still reduceByKey is better cause
            // reduceByKey has two stages
            // at first stage : it will apply the reduce function on each partition which does NOT reduce any shuffling.
            // at second stage : it will shuffle to bring similar keys into one partition but this shuffling is very minimal.

            // reduceByKey is sometimes called as 'map-side-reduce'.
        }
    }
}
