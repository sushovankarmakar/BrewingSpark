package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import udemy.virtualPairProgrammers.Util;

import java.util.Arrays;

public class _7_Practical_KeyWordRanking {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("keyword-ranking")
                .setMaster("local[*]");

        try (JavaSparkContext jsc = new JavaSparkContext(sparkConf)) {

            // Steps
            // 1. Load the file into RDD -> textFile()
            // 2. Filtering all noisy lines i.e., timestamps, empty lines and turning all of them in lowercase -> map()
            // 3. Converting lines into words -> flatmap()
            // 4. Filtering out the empty and boring words -> filter()
            // 5. Count up the remaining words -> reduceByKey()
            // 6. Find the top 10 most frequently used words -> mapToPair(), sortByKey(), take()

            String inputFileDockerCourseSubtitles = "src/main/resources/udemy/virtualPairProgrammers/subtitles/input-docker-course.txt";
            String inputFileSpringCourseSubtitles = "src/main/resources/udemy/virtualPairProgrammers/subtitles/input-spring-course.txt";

            JavaPairRDD<Long, String> wordFrequencies = jsc.textFile(inputFileSpringCourseSubtitles)
                    .map(line -> line.replaceAll("[^a-zA-Z\\s]", "").trim().toLowerCase())
                    .flatMap(line -> Arrays.asList(line.split(" ")).iterator())
                    .filter(word -> !word.isEmpty() && Util.isNotBoring(word))
                    .mapToPair(word -> new Tuple2<>(word, 1L)) // take PairFunction
                    .reduceByKey(Long::sum) // takes Function2
                    .mapToPair(pair -> new Tuple2<>(pair._2, pair._1))  // switching value and key so that we can sort by value
                    .sortByKey(false); // sorting in descending order

            // to get the output in sorted order, we need to call take() function, not foreach function.
            wordFrequencies.take(10)
                    .forEach(System.out::println);

            System.out.println("There are " + wordFrequencies.getNumPartitions() + " partitions. ");

            // Q: why do sorts not work with foreach in Spark ?
            // because: driver is sending foreach function to each partition and this function will be executed in parallel in each partition

            // Now, although we're running in a single machine, this single machine has multiple cores
            // and the default configuration of spark is that
            // for each core, java will spin up a new thread

            // so here, the output we're seeing is as simple as we would run a multithreaded java program
            // each thread is running on each partition and printing some lines
            // before another thread is interrupting the current thread and then starts its processing.

            // Below code will give us a wrong result, uncomment this to understand why sorting not works with foreach
            // wordFrequencies.coalesce(1).foreach(element -> System.out.println(element));
        }
    }
}
