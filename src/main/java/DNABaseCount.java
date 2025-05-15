import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class DNABaseCount
{
    public static void main(String[] args)
    {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("app");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        JavaRDD<String> inputRDD = jsc.textFile("C:\\Users\\Lenovo\\Desktop\\log_.txt");

        JavaRDD<Map<Character, Long>> mapJavaRDD = inputRDD.mapPartitions(new FlatMapFunction<Iterator<String>, Map<Character, Long>>() {
            Map<Character, Long> baseCounts = new HashMap<Character, Long>();

            @Override
            public Iterator<Map<Character, Long>> call(Iterator<String> iterator) throws Exception {
                while (iterator.hasNext()) {
                    String value = iterator.next();
                    if (value.startsWith(">")) {
                        // do nothing
                    } else {
                        String str = value.toUpperCase();
                        for (int i = 0; i < str.length(); ++i) {
                            char c = str.charAt(i);
                            Long count = baseCounts.get(c);
                            if (count == null) {
                                baseCounts.put(c, 1L);
                            } else {
                                baseCounts.put(c, count + 1L);
                            }
                        }
                    }
                }
                return Collections.singletonList(baseCounts).iterator();
            }
        });
        List<Map<Character, Long>> collect = mapJavaRDD.collect();
    }

}