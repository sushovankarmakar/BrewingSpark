package udemy.virtualPairProgrammers.sparkML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class _1_GymCompetitors {

    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf sparkConf = new SparkConf()
                .setAppName("spark-ml-1")
                .setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder()
                .config(sparkConf)
                .getOrCreate();

        Dataset<Row> csvDataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/GymCompetition.csv");

        csvDataset.printSchema();
        csvDataset.show(false);

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"Age", "Height", "Weight"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvDataset);
        Dataset<Row> modelInputData = csvDataWithFeatures
                .select("NoOfReps", "features")
                .withColumnRenamed("NoOfReps", "label");

        modelInputData.show(false);

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel linearRegressionModel = linearRegression.fit(modelInputData);
        double intercept = linearRegressionModel.intercept();
        Vector coefficients = linearRegressionModel.coefficients();
        System.out.println("The model has intercept " + intercept + " and coefficients " + coefficients);

        linearRegressionModel.transform(modelInputData).show(false);

    }
}
