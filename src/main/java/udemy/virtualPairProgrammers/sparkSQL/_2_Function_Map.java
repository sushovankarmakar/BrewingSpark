package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.ArrayList;
import java.util.List;

public class _2_Function_Map {

    public static void main(String[] args) {

        List<Integer> inputData = new ArrayList<>();
        inputData.add(353);
        inputData.add(12345);
        inputData.add(93);
        inputData.add(38);
        inputData.add(623);
        inputData.add(117);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("startingSpark")
                .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        JavaRDD<Integer> myRDD = sc.parallelize(inputData);

        //JavaRDD<Double> squareRoots = myRDD.map((value) -> Math.sqrt(value));
        JavaRDD<Double> squareRoots = myRDD.map(Math::sqrt); // more modern syntax, when we're passing one value and operating on one value only, then we can use this modern syntax
        squareRoots.foreach(val -> System.out.println("Sqrt val : " + val));

        //squareRoots.foreach(System.out::println); // more modern syntax - this will give me 'not serializable exception' - because before sending println function to RDDs, spark tries to serialize it but println function in java is NOT serializable.
        squareRoots.collect().forEach(System.out::println); // this is a fix of the above line.

        System.out.println(squareRoots.toDebugString());

        // ------------------------------------------------------------------------------------------------------------
        // how many elements in RDD ? - two ways we can do
        // 1. to count the number of items inside the RDD, we can use .count() operation, but it should be use if we're at the end of the process
        System.out.println(squareRoots.count());

        // 2. we can use map and reduce
        //Long totalNumOfElements = squareRoots.map(val -> 1L).reduce((val1, val2) -> val1 + val2); // use long type instead of integer, cause in big data, integer can be max 2 billion.
        Long totalNumOfElements = squareRoots.map(val -> 1L).reduce(Long::sum); // more modern syntax
        System.out.println("Total number of elements are : " + totalNumOfElements);

        sc.close();
    }
}
