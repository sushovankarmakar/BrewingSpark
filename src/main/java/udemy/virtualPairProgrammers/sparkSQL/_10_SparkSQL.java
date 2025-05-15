package udemy.virtualPairProgrammers.sparkSQL;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Scanner;

import static org.apache.spark.sql.functions.col;

public class _10_SparkSQL {

    public static void main(String[] args) {

        String inputFilePath = "src/main/resources/udemy/virtualPairProgrammers/exams/students.csv";

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-sql")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();
        // config("spark.sql.warehouse.dir", "file:///c:/tmp/") -> here spark will store some files related to spark sql

        // Dataset is an abstraction that represents our data
        // we're still working with RDD objects; these objects are hidden into the Dataset object.
        Dataset<Row> inputData = sparkSession.read()
                .option("header", "true")
                .csv(inputFilePath);

        inputData.show();

        long totalCount = inputData.count();
        System.out.println("Total count : " + totalCount); // this is a distributed count, spread across multiple nodes

        // ------------------------------- Access column values -------------------------------
        accessColumnValues(inputData);

        // ------------------------------- Filter using sql expression -------------------------------
        filterUsingSqlExpression(inputData);

        // ------------------------------- Filter using lambda function -------------------------------
        filterUsingLambdaFunction(inputData);

        // ------------------------------- Filter using column class -------------------------------
        // --------------------------------------------------------- version 1
        filterUsingColumnClass(inputData);
        // --------------------------------------------------------- version 2
        filterUsingFunctionCol(inputData);

        // -------------------------------  Spark Temporary View -------------------------------
        sparkTemporaryView(inputData, sparkSession);

        Scanner sc = new Scanner(System.in);
        sc.nextLine();

        sparkSession.cloneSession();
    }

    private static void accessColumnValues(Dataset<Row> inputData) {
        // if we're reading data in csv format, then every column value is raw string, we've to convert it into various datatypes
        Row firstRow = inputData.first();
        String firstSubject = firstRow.get(2).toString();
        System.out.println("First subject is : " + firstSubject);

        int firstYear = Integer.parseInt(firstRow.getAs("year"));
        System.out.println("First year is : " + firstYear);
    }

    private static void filterUsingSqlExpression(Dataset<Row> inputData) {
        System.out.println("Filter using sql expression");
        Dataset<Row> modernArtResults = inputData.filter("subject = 'Modern Art' AND year >= 2007 ");
        modernArtResults.show(5);
    }

    private static void filterUsingLambdaFunction(Dataset<Row> inputData) {
        System.out.println("Filter using lambda function");
        Dataset<Row> moderArtResults1 = inputData.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                return row.getAs("subject").equals("Modern Art")
                        && Integer.parseInt(row.getAs("year")) >= 2007;
            }
        });
        moderArtResults1.show(5);
    }

    private static void filterUsingColumnClass(Dataset<Row> inputData) {
        System.out.println("Filter using column class : version 1");
        Column subject = inputData.col("subject");
        Column year = inputData.col("year");

        Dataset<Row> modernArtResults2 = inputData.filter(subject.equalTo("Modern Art")
                .and(year.geq(2007))
        );
        modernArtResults2.show(5);
    }

    private static void filterUsingFunctionCol(Dataset<Row> inputData) {
        System.out.println("Filter using column class : version 2");
        Column subject = col("subject");
        Column year = col("year");

        Dataset<Row> modernArtResults = inputData.filter(subject.equalTo("Modern Art")
                .and(year.geq(2007))
        );
        modernArtResults.show(5);
    }

    private static void sparkTemporaryView(Dataset<Row> inputData, SparkSession sparkSession) {
        System.out.println("Using spark temporary view");

        inputData.createOrReplaceTempView("students");
        Dataset<Row> modernArtResults = sparkSession.sql("SELECT * FROM students s WHERE s.subject = 'Modern Art' AND s.year >= 2007");
        modernArtResults.show(5);

        // here we're calling distinct, shuffle will happen.
        Dataset<Row> distinctYears = sparkSession.sql("SELECT distinct year FROM students s");
        distinctYears.show();
    }
}
