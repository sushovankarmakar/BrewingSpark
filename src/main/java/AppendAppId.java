import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class AppendAppId {
    public static void main(String[] args) {
        // Create a SparkSession
        SparkSession sparkSession = SparkSession.builder()
                .master("local")
                .appName("example")
                .getOrCreate();

        // Set the output directory and file name prefix
        String outputDir = "/test-output/append-app-id";
        String filePrefix = "mydata";

        // Set the configuration properties
        SparkConf sparkConf = sparkSession.sparkContext().getConf();
        sparkConf.set("spark.hadoop.mapreduce.output.fileoutputformat.outputdir", outputDir);
        String applicationId = sparkSession.sparkContext().applicationId();
        sparkConf.set("spark.hadoop.mapreduce.output.fileoutputformat.output.filenameextension", "-sushovan-" + applicationId);

        // Write the data to the output directory with the application ID appended to the filename
        Dataset<Row> data = sparkSession.read()
                .format("text")
                .option("header", false)
                .load("/example_files");

        System.out.println("App id " + applicationId);
        data.write()
                .mode("overwrite")
                .option("header", "true")
                .csv(outputDir + "/" + filePrefix); // outputDir + "/" + filePrefix

        // Stop the SparkSession
        sparkSession.stop();
    }
}