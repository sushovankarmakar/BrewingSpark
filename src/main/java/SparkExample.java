import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class SparkExample {

    public static void main(String[] args) {
        // Initialize Spark session
        SparkSession spark = SparkSession.builder()
                .appName("Spark Example")
                .config("spark.master", "local")
                .getOrCreate();

        // Read JSON file into a DataFrame
        Dataset<Row> df = spark.read().json("path/to/your/json/file.json");

        // Show the content of the DataFrame
        df.show();

        // Perform some transformations
        df.printSchema();
        df.select("name").show();
        df.filter(df.col("age").gt(21)).show();

        // Stop the Spark session
        spark.stop();
    }
}
