import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Main {

    /**
     * https://www.youtube.com/watch?v=yztRtbup6Pc
     */
    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        List<Double> inputs = new ArrayList<>();
        inputs.add(35.5);
        inputs.add(12.994);
        inputs.add(90.32);
        inputs.add(20.32);

        SparkConf conf = new SparkConf()
                .setAppName("starting-spark")
                .setMaster("local[*]");

        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        //JavaRDD<Double> javaRDD = sparkContext.parallelize(inputs);

        String filePath = "/development/src/main/resources/content.txt";

        JavaRDD<String> textFile = sparkContext.textFile(filePath);

        //textFile.filter((Function<String, Boolean>) s -> null);

        /*JavaRDD<String> filteredLines = textFile.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String line) throws Exception {
                return line.contains("Spark");
            }
        });*/


        JavaRDD<String> filteredLines = textFile.filter(line -> line.contains("Spark"));
        //System.out.println(filteredLines.count());
        //System.out.println(filteredLines.first());


        JavaRDD<String> inputFile = sparkContext.textFile(filePath);
        JavaRDD<String> wordsFromFile = inputFile.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterator<String> call(String s) throws Exception {
                return null;
            }
        });

        //JavaRDD<String> lines = sparkContext.textFile("hdfs://log.txt");
        //lines.flatMap(line -> line.split(" ")));

        // Map each line to multiple words
        /*JavaRDD<String> words = textFile.flatMap(
                new FlatMapFunction<String, String>() {
                    public Iterator<String> call(String line) {
                        List<String> list = Arrays.asList(line.split(" "));
                        return list.iterator();
                    }
                });*/
        JavaRDD<String> words = textFile.flatMap(line -> Arrays.asList(line.split(" ")).iterator());

        System.out.println(words.first());


        sparkContext.close();
    }
}
