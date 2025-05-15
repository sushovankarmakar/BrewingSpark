package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.Optional;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class _8_Joins {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("join-example")
                .setMaster("local[*]");

        List<Tuple2<Integer, Integer>> userVisitRaw = new ArrayList<>();
        userVisitRaw.add(new Tuple2<>(4, 18));
        userVisitRaw.add(new Tuple2<>(6, 4));
        userVisitRaw.add(new Tuple2<>(10, 9));

        List<Tuple2<Integer, String>> usersRaw = new ArrayList<>();
        usersRaw.add(new Tuple2<>(1, "John"));
        usersRaw.add(new Tuple2<>(2, "Bob"));
        usersRaw.add(new Tuple2<>(3, "Alan"));
        usersRaw.add(new Tuple2<>(4, "Doris"));
        usersRaw.add(new Tuple2<>(5, "Marybelle"));
        usersRaw.add(new Tuple2<>(6, "Raquel"));

        try (JavaSparkContext jsc = new JavaSparkContext(sparkConf)) {

            JavaPairRDD<Integer, Integer> userVisits = jsc.parallelizePairs(userVisitRaw);
            JavaPairRDD<Integer, String> users = jsc.parallelizePairs(usersRaw);

            System.out.println("Inner join");
            JavaPairRDD<Integer, Tuple2<Integer, String>> innerJoin = userVisits.join(users);

            innerJoin.foreach(val -> System.out.println(val + " "));
            System.out.println(" -------- ");



            System.out.println("Left outer join");
            JavaPairRDD<Integer, Tuple2<Integer, Optional<String>>> leftOuterJoin = userVisits.leftOuterJoin(users);

            leftOuterJoin.foreach(val -> System.out.println(val + " "));
            System.out.println(" -------- ");



            System.out.println("Right outer join");
            JavaPairRDD<Integer, Tuple2<Optional<Integer>, String>> rightOuterJoin = userVisits.rightOuterJoin(users);

            rightOuterJoin.foreach(val -> System.out.println(val + " "));
            System.out.println(" -------- ");



            System.out.println("Full outer join");
            JavaPairRDD<Integer, Tuple2<Optional<Integer>, Optional<String>>> fullOuterJoin = userVisits.fullOuterJoin(users);

            fullOuterJoin.foreach(val -> System.out.println(val + " "));
            System.out.println(" -------- ");



            System.out.println("Cross Join or Cartesian");
            JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Integer, String>> cartesian = userVisits.cartesian(users);

            cartesian.foreach(val -> System.out.println(val + " "));
            System.out.println(" -------- ");
        }
    }
}
