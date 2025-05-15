package udemy.virtualPairProgrammers.sparkSQL;

import com.google.common.collect.Iterables;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class _4_PairRDD_WithFluentAPI {

    // https://www.databricks.com/blog/2014/04/14/spark-with-java-8.html

    public static void main(String[] args) {

        List<String> inputData = new ArrayList<>();
        inputData.add("WARN: Tuesday 4 September 0405");
        inputData.add("ERROR: Tuesday 4 September 040");
        inputData.add("FATAL: Wednesday 5 September 0405");
        inputData.add("ERROR: Tuesday 4 September 0405");
        inputData.add("WARN: Tuesday 4 September 0405");

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("startingSpark")
                .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // fluent api: one function's output can be another function's input. so we can do chaining.

        sc.parallelize(inputData)
                .mapToPair(rawValue -> new Tuple2<>(rawValue.split(":")[0], 1L))
                .reduceByKey(Long::sum)
                .foreach(tuple ->
                        System.out.println(tuple._1 + " has " + tuple._2 + " records")
                );

        // groupByKey example:
        // warning : group by key causes serious performance issues, data skewness issues
        // also, the value is gives for each key is Iterable type. We can't call .size() method on Iterable type.
        sc.parallelize(inputData)
                .mapToPair(rawValue -> new Tuple2<>(rawValue.split(":")[0], 1L))
                .groupByKey().foreach(tuple -> System.out.println(tuple._1 + " has " + Iterables.size(tuple._2) + " records"));

        sc.close();
    }
}
