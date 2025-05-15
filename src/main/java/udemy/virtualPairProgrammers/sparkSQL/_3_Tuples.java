package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import udemy.virtualPairProgrammers.IntegerWithSquareRoot;

import java.util.ArrayList;
import java.util.List;

public class _3_Tuples {

    public static void main(String[] args) {

        List<Integer> inputData = new ArrayList<>();
        inputData.add(35);
        inputData.add(12);
        inputData.add(9);
        inputData.add(3);
        inputData.add(6);
        inputData.add(11);

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("startingSpark")
                .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<Integer> integerRDD = sc.parallelize(inputData);
        //JavaRDD<IntegerWithSquareRoot> sqrtRdd = integerRDD.map(val -> new IntegerWithSquareRoot(val)); // old syntax
        JavaRDD<IntegerWithSquareRoot> sqrtRdd = integerRDD.map(IntegerWithSquareRoot::new); // new syntax

        JavaRDD<Tuple2<Integer, Double>> tupleRdd = integerRDD.map(val -> new Tuple2<>(val, Math.sqrt(val))); // Tuple22 is the limit

        sc.close();
    }
}
