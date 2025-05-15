package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class _5_FlatMaps {

    public static void main(String[] args) {

        List<String> inputData = new ArrayList<>();
        inputData.add("WARN: Tuesday 4 September 0405");
        inputData.add("ERROR: Tuesday 4 September 040");
        inputData.add("FATAL: Wednesday 5 September 0405");
        inputData.add("ERROR: Tuesday 4 September 0405");
        inputData.add("WARN: Tuesday 4 September 0405");

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("flatMaps-examples")
                .setMaster("local[*]");

        try (JavaSparkContext jsc = new JavaSparkContext(sparkConf)) {

            JavaRDD<String> inputRdd = jsc.parallelize(inputData);
            JavaRDD<String> words = inputRdd.flatMap(val ->
                    Arrays.asList(val.split(" ")).iterator()
            );

            // filtering out numeric words
            JavaRDD<String> filteredWords = words.filter(word -> !word.matches("\\d+"));

            filteredWords.foreach(word ->  System.out.println(word + " "));
        }
    }
}
