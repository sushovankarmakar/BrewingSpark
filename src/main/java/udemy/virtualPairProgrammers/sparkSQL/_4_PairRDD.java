package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class _4_PairRDD {

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

        JavaRDD<String> originalLogMsg = sc.parallelize(inputData);

        JavaPairRDD<String, String> levelAndDatePair = originalLogMsg.mapToPair(rawValue -> {
            String[] columns = rawValue.split(":");
            String level = columns[0];
            String date = columns[1];
            return new Tuple2<>(level, date);
        });

        // group by key causes serious performance issues, data skewness issues

        JavaPairRDD<String, Long> levelPair = originalLogMsg.mapToPair(rawValue -> { // keeping long instead of int allows us to work in trillions of records
            String[] columns = rawValue.split(":");
            String level = columns[0];
            return new Tuple2<>(level, 1L);
        });

        //levelPair.reduceByKey((val1, val2) -> val1 + val2); - old syntax
        JavaPairRDD<String, Long> levelAndCounts = levelPair.reduceByKey(Long::sum);// new syntax

        System.out.println(levelAndCounts.toDebugString());

        levelAndCounts.foreach(tuple -> System.out.println(tuple._1 + " has " + tuple._2 + " records"));

        sc.close();
    }
}
