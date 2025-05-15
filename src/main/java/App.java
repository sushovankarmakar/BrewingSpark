import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import scala.Char;
import scala.Tuple2;

import java.util.*;

public class App {
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setMaster("local").setAppName("app");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        JavaRDD<String> inputRDD = jsc.textFile("C:\\Users\\Lenovo\\Desktop\\log_.txt");
        
        JavaRDD<String> javaRDD = inputRDD.flatMap(e -> Arrays.asList(e.split(" ")).iterator());
        //List<String> collect1 = javaRDD.collect();
        JavaRDD<String> map = javaRDD.map(s -> s.toLowerCase(Locale.ROOT));
        //List<String> collect2 = map.collect();
        JavaPairRDD<String, Integer> flatMapPairRDD = map.flatMapToPair(new PairFlatMapFunction<String, String, Integer>() {
            @Override
            public Iterator<Tuple2<String, Integer>> call(String s) throws Exception {
                List<Tuple2<String, Integer>> list = new ArrayList<>();
                if (s.contains(">")) {
                    list.add(new Tuple2<>("z", 1));
                } else {
                    String[] str = s.split("");
                    for (String c : str) {
                        list.add(new Tuple2<>(c, 1));
                    }
                }
                return list.iterator();
            }
        });
        List<Tuple2<String, Integer>> collect = flatMapPairRDD.collect();
        JavaPairRDD<String, Integer> stringIntegerJavaPairRDD = flatMapPairRDD.reduceByKey((x, y) -> x + y);
        System.out.println(stringIntegerJavaPairRDD.collect());
    }
}