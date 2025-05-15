import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

/**
 * https://github.com/jbbarquero/spark-examples/blob/master/src/main/java/com/malsolo/spark/examples/WordCount.java
 * https://spark.apache.org/docs/2.0.1/programming-guide.html
 * https://stackoverflow.com/a/26000677
 *
 * https://github.com/apache/spark/blob/master/examples/src/main/java/org/apache/spark/examples/JavaWordCount.java
 */
public class WordCount {

    private static final String INPUT_FILE_TEXT =
            "src/main/resources/the_constitution_of_the_united_states.txt";
    private static final String OUTPUT_FILE_TEXT = "out";
    private static final Pattern SPACE = Pattern.compile(" ");

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession session = SparkSession.builder()
                .appName("Word Count with Spark")
                .master("local")
                .getOrCreate();

        //SparkConf conf = new SparkConf().setMaster("local").setAppName("Word Count with Spark");

        JavaSparkContext context = new JavaSparkContext(session.sparkContext());

        //context.sc().longAccumulator();

        JavaRDD<String> lines = context.textFile(INPUT_FILE_TEXT);

        LongAccumulator blankLines = session.sparkContext().longAccumulator();
        Broadcast<List<String>> wordsToIgnore = context.broadcast(getWordsToIgnore());

        JavaPairRDD<String, Integer> counts =
                lines.flatMap(line -> {
                            if ("".equals(line)) {
                                blankLines.add(1);
                            }
                            return Arrays.asList(line.trim().split(SPACE.toString())).iterator();
                        })
                        .filter(word -> !wordsToIgnore.value().contains(word))
                        .mapToPair(word -> new Tuple2<>(word, 1))
                        .reduceByKey((x, y) -> x + y)
                        .sortByKey();

        //System.out.println(filter.count());

        if (!Files.exists(Paths.get(OUTPUT_FILE_TEXT))) {
            counts.saveAsTextFile(OUTPUT_FILE_TEXT);
        }

        System.out.println("Number of words : " + counts.count());
        System.out.println("Total number of blank lines : " + blankLines.value());

        context.close();
    }

    private static List<String> getWordsToIgnore() {
        return Arrays.asList("the", "of", "and", "for");
    }
}
