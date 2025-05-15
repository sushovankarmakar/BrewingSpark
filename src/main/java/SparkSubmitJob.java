import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.sql.SparkSession;

import java.util.Arrays;
import java.util.List;

public class SparkSubmitJob {

    /**
     * https://www.youtube.com/watch?v=cu2E0sSlWsY (Java in Spark | Spark-Submit Job with Spark UI | Tech Primers)
     */
    public static void main(String[] args) {
        SparkSession session = SparkSession.builder()
                .appName("SparkSubmitJob")
                .master("local[3]")
                .getOrCreate();

        try (JavaSparkContext context = new JavaSparkContext(session.sparkContext())) {

            List<Integer> integers = Arrays.asList(1, 4, 34, 56, 9, 0, 23, 71, 14, 60, 44, 27, 3, 78, 45, 28);

            // num of slices indicates the number of executors. here it is 3 executors.
            JavaRDD<Integer> integerRDD = context.parallelize(integers, 3);

            /*integerRDD.foreach(new VoidFunction<Integer>() {
                @Override
                public void call(Integer integer) throws Exception {
                    System.out.println("Number : " + integer);
                }
            });*/

            integerRDD.foreach(integer -> {
                Thread.sleep(3000);
                System.out.println("Number : " + integer);
            });

            Thread.sleep(100000);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
