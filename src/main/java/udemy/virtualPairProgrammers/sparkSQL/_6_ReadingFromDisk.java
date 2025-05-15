package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;

public class _6_ReadingFromDisk {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]")
                .setAppName("reading-from-disk");

        try (JavaSparkContext jsc = new JavaSparkContext(sparkConf)) {

            // in big data, we usually read from a distributed file system like s3 or hadoop
            // when we call textFile; it doesn't store the entire file into the memory

            // textFile() asked driver to inform its workers to load parts (partitions) of this input file
            JavaRDD<String> inputRDD = jsc.textFile("src/main/resources/udemy/virtualPairProgrammers/subtitles/input-docker-course.txt");

            JavaRDD<String> words = inputRDD.flatMap(line -> Arrays.asList(line.split(" ")).iterator());

            words.foreach(word -> System.out.println(word + " "));
        }
    }
}
