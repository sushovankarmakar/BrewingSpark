package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.ArrayList;
import java.util.List;

public class _1_Function_Reduce {

    public static void main(String[] args) {

        List<Double> inputData = new ArrayList<>();
        inputData.add(35.3);
        inputData.add(12.345);
        inputData.add(9.3);
        inputData.add(3.8);
        inputData.add(6.23);
        inputData.add(11.7);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("startingSpark")
                .setMaster("local[*]"); // setting up in local configuration as we don't have the cluster.
                // local[*] means 'use all the cores available in this machine'
                // local means 'running in a single thread' - less performance

        JavaSparkContext sc = new JavaSparkContext(sparkConf);  // JavaSparkContext is a connection to the Spark cluster

        // JavaRDD : a bridge between java and spark which is written in Scala. So under the hood, JavaRDD is a wrapper which internally connect with the Scala RDD
        // parallelize : loading a java collection and turn it into a RDD
        JavaRDD<Double> myRDD = sc.parallelize(inputData);

        System.out.println(myRDD.toDebugString());

        Double sum = myRDD.reduce((value1, value2) -> value1 + value2);// reduce action contains shuffle operation also.
        System.out.println("(Using lamda) Sum is : " + sum);

        Double sum1 = myRDD.reduce(Double::sum);
        System.out.println("(Using method reference) Sum is : " + sum1);

        // Minimum value in an Array
        Double minValue = myRDD.reduce((val1, val2) -> Math.min(val1, val2));
        System.out.println("Min value " + minValue);

        // Maximum value in an array
        Double maxValue = myRDD.reduce((val1, val2) -> Math.max(val1, val2));
        System.out.println("Max value " + maxValue);

        // average of all the values in an array
        Double sum2 = myRDD.reduce(Double::sum);
        Long nums = myRDD.map(val -> 1L).reduce(Long::sum);
        System.out.println("Avg is : " + sum2/nums);

        sc.close();
    }
}
