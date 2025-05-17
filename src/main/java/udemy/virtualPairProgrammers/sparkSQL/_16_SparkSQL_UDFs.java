package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.arrow.flatbuf.Bool;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class _16_SparkSQL_UDFs {

    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-udf")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        Dataset<Row> inputData = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/udemy/virtualPairProgrammers/exams/students.csv");

        //inputData = inputData.withColumn("pass", functions.lit("YES"));
        //inputData = inputData.withColumn("pass", functions.col("grade").equalTo("A+"));

        // UDF allows us to add our own functions into the Spark API.
        // syntax:
        // register(name, lambda function, return type)

        sparkSession.udf().register("hasPassed", (String grade, String subject) -> {
                    if (subject.equals("Biology")) {
                        return (grade.startsWith("A"));
                    }
                    return grade.startsWith("A") || grade.startsWith("B") || grade.startsWith("C");
                },
                DataTypes.BooleanType);

        sparkSession.udf().register("hasPassed1",
                hasPassedFunction,
                DataTypes.BooleanType);

        inputData = inputData.withColumn("pass", functions.callUDF(
                "hasPassed",
                functions.col("grade"),
                functions.col("subject")
        ));

        inputData.show(false);
    }


    // this is just for reference. favor Java 8 lambdas
    private static final UDF2<String, String, Boolean> hasPassedFunction = new UDF2<String, String, Boolean>() {
        @Override
        public Boolean call(String grade, String subject) {
            if (subject.equals("Biology")) {
                return grade.startsWith("A");
            }
            return grade.startsWith("A") || grade.startsWith("B") || grade.startsWith("C");
        }
    };
}
